"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ctypes
import functools
from dataclasses import dataclass
from math import prod
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Sequence

import cutlass
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass._mlir.ir as ir
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Pointer, Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.base_dsl.dsl import extract_mlir_values
from cutlass._mlir.dialects import scf
from cutlass._mlir.dialects import llvm

import torch
from torch.distributed import ProcessGroup

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .nvshmem import get_nvshmem_module

def as_tensor(pointer, shape, torch_type):
    if torch_type.itemsize == 1:
        cytype = ctypes.c_uint8
    elif torch_type.itemsize == 2:
        cytype = ctypes.c_uint16
    elif torch_type.itemsize == 4:
        cytype = ctypes.c_uint32
    elif torch_type.itemsize == 8:
        cytype = ctypes.c_uint64
    else:
        raise ValueError(f'Unsupported torch dtype: {torch_type}')
    cpointer = ctypes.cast(pointer, ctypes.POINTER(cytype))
    arr = (cpointer._type_ * prod(shape)).from_address(
        ctypes.addressof(cpointer.contents))
    return torch.frombuffer(arr, dtype=torch_type).view(*shape)

@dsl_user_op
def multimem_ld_reduce(
    mc_ptr: Pointer,
    *,
    loc=None,
    ip=None,
):
    # ld reduce 8x f16 elts
    mc_ptr_int = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    i32 = ir.IntegerType.get_signless(32)
    return_struct = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [mc_ptr_int],
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
    )
    return_regs = [
          llvm.extractvalue(i32, return_struct, [i]) for i in range(4)
    ]
    return return_regs[0], return_regs[1], return_regs[2], return_regs[3]


@dsl_user_op
def multimem_st(
    mc_ptr: Pointer,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc=None,
    ip=None,
):
    # st 8x f16 elts
    mc_ptr_int = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    i32 = ir.IntegerType.get_signless(32)
    llvm.inline_asm(
        i32,
        [mc_ptr_int, x, y, z, w],
        "multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
    )

@dsl_user_op
def signal_multimem(
    flag_mc,
    is_relaxed=False,
    *,
    loc=None,
    ip=None,
):
    mode = "relaxed" if is_relaxed else "release"
    flag_ptr_int = flag_mc.toint().ir_value()
    llvm.inline_asm(
        None,
        [flag_ptr_int],
        f"""
        {{
            multimem.red.{mode}.sys.global.add.u32 [$0], 1;
            fence.proxy.alias;
        }}""",
        "l",
        has_side_effects=True,
        asm_dialect=0,
    )

@dsl_user_op
def wait_loop(
    flag,
    num_ranks,
    is_relaxed=False,
    *,
    loc=None,
    ip=None,
):
    mode = "relaxed" if is_relaxed else "acquire"
    flag_ptr_int = flag.toint().ir_value()
    llvm.inline_asm(
        None,
        [flag_ptr_int, num_ranks.ir_value()],
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;
            wait_signal:
                atom.global.sys.{mode}.cas.b32 %tmp32_0, [$0], $1, 0;
                setp.eq.u32 %p0, %tmp32_0, $1;
                @!%p0 bra wait_signal;
        }}""",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
    )

class MultimemAllReduce:

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        max_buffer_elements: int,
        dtype: torch.dtype,
        device: torch.device,
        group: Optional[ProcessGroup] = None,
        should_init: bool = True,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.dtype = dtype
        self.device = device
        self.max_buffer_elements = max_buffer_elements
        self.group = group
        self.nvshmem_module = get_nvshmem_module()

        # TODO(asamani): make this configurable
        tensor_dtype = cutlass.Float16
        self.num_warps_per_block = 8 # for fp16 and bf16
        self.num_ctas = 128 # max is the number of sms
        self.tensor_size = max_buffer_elements
        numel_per_thread = 8

        max_work_per_iter = numel_per_thread * self.num_warps_per_block * 32 * self.num_ctas * world_size
        self.num_iter_per_thread = self.tensor_size//max_work_per_iter

        # TODO(asamani): check if this configuratio works with impl
        self.should_init = should_init
        if self.should_init:
            self.init_nvshmem()

        # assert PE and world size match
        my_pe = self.nvshmem_module.nvshmem_my_pe()
        n_pes = self.nvshmem_module.nvshmem_n_pes()
        if my_pe != local_rank:
            raise RuntimeError(f"Rank {local_rank}: PE mismatch! Expected PE {local_rank}, got PE {my_pe}")
        if n_pes != world_size:
            raise RuntimeError(f"Rank {local_rank}: World size mismatch! Expected {world_size}, got {n_pes}")

        self.cute_tensor_barrier_start, self.nvshmem_tensor_barrier_start, self.tensor_mc_memerf_barrier_start = self.create_barrier_flags([self.num_ctas], cutlass.Int32, device)

        self.cute_tensor_barrier_end, self.nvshmem_tensor_barrier_end, self.tensor_mc_memerf_barrier_end = self.create_barrier_flags([self.num_ctas], cutlass.Int32, device)

        self.input_tensor, self.input_torch, self.input_mc_memref  = self.create_and_permute_tensor(
            [self.tensor_size],
            dtype=tensor_dtype,
            device=device,
            is_mc=True,
            fill_value=None,
        )
        self.output_tensor, self.output_torch, self.output_mc_memref  = self.create_and_permute_tensor(
            [self.tensor_size],
            dtype=tensor_dtype,
            device=device,
            is_mc=True,
            fill_value=0,
        )

        torch.distributed.barrier(self.group)
    
    def init_nvshmem(self):
        if self.local_rank == 0:
            uid = self.nvshmem_module.nvshmem_get_unique_id()
        else:
            uid = torch.zeros(self.nvshmem_module.nvshmem_unique_id_size(), dtype=torch.uint8, device="cpu")
        torch.distributed.broadcast(uid, src=0)
        torch.distributed.barrier(self.group)
        init_status = self.nvshmem_module.nvshmem_init(uid, self.local_rank, self.world_size)
        torch.cuda.synchronize()
        if init_status != 0:
            raise RuntimeError("Failed to initialize nvshmem")
    
    @cute.kernel
    def allreduce_kernel(
        self,
        gInput: cute.Tensor,
        gOutput: cute.Tensor,
        gInputMC: cute.Tensor,
        gOutputMC: cute.Tensor,
        total_elements: cutlass.Constexpr[int],
        rank_id: cutlass.Constexpr[int],
        red_elements: cutlass.Constexpr[int],
        num_iter_per_thread: cutlass.Constexpr[int],
        cute_tensor_barrier_start: cute.Tensor,
        cute_tensor_barrier_end: cute.Tensor,
        tensor_mc_memerf_barrier_start: cute.Tensor,
        tensor_mc_memerf_barrier_end: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        gdim, _, _ = cute.arch.grid_dim()

        thread_idx = bidx * bdim + tidx
        # sync before start
        #cute.arch.sync_warp()
        if tidx == 0: # run once per block
            flag_start_mc = tensor_mc_memerf_barrier_start.iterator + bidx
            flag_start_uc = cute_tensor_barrier_start.iterator + bidx
            signal_multimem(flag_start_mc, is_relaxed=True)
            wait_loop(flag_start_uc, Int32(self.world_size), is_relaxed=True)
        # reduction loop
        cute.arch.sync_threads()
        #cute.arch.sync_warp()
        for i in cutlass.range_dynamic(num_iter_per_thread):
            global_offset = rank_id * gdim * bdim * red_elements * num_iter_per_thread
            device_offset = thread_idx * red_elements
            iter_offset = i * red_elements * bdim * gdim
            offset = global_offset + device_offset + iter_offset
            elem_coords = (offset,)
            idx = cute.crd2idx(elem_coords, gOutputMC.layout)
            mc_ptr_inp = gInputMC.iterator + idx
            mc_ptr = gOutputMC.iterator + idx
            x, y, z, w = multimem_ld_reduce(mc_ptr_inp)
            multimem_st(mc_ptr, x, y, z, w)

        # sync before exiting the kernel
        cute.arch.sync_threads()
        #cute.arch.sync_warp()
        if tidx == 0: # run once per block
            flag_end_mc = tensor_mc_memerf_barrier_end.iterator + bidx
            flag_end_uc = cute_tensor_barrier_end.iterator + bidx
            signal_multimem(flag_end_mc, is_relaxed=False)
            wait_loop(flag_end_uc, Int32(self.world_size), is_relaxed=False)

    @cute.jit
    def all_reduce_jitted(
        self,
        gInput: cute.Tensor,
        gOutput: cute.Tensor,
        gInputMC: cute.Tensor,
        gOutputMC: cute.Tensor,
        num_warps_per_block: cutlass.Constexpr[int],
        num_ctas: cutlass.Constexpr[int],
        num_iter_per_thread: cutlass.Constexpr[int],
        rank_id: cutlass.Constexpr[int],
        num_ranks: cutlass.Constexpr[int],
        cute_tensor_barrier_start: cute.Tensor,
        cute_tensor_barrier_end: cute.Tensor,
        tensor_mc_memerf_barrier_start: cute.Tensor,
        tensor_mc_memerf_barrier_end: cute.Tensor,
    ):
        total_elements = cute.size(gInput)
        numel_per_thread = 8 # for fp16 and bf16
        kernel = self.allreduce_kernel(
            gInput, 
            gOutput, 
            gInputMC, 
            gOutputMC, 
            total_elements, 
            rank_id, 
            numel_per_thread, 
            num_iter_per_thread,
            cute_tensor_barrier_start,
            cute_tensor_barrier_end,
            tensor_mc_memerf_barrier_start,
            tensor_mc_memerf_barrier_end,
        )
        kernel.launch(grid=(num_ctas, 1, 1),
                    block=(num_warps_per_block * 32, 1, 1))

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        numel = inp.numel()
        input_buffer = self.input_torch.narrow(0, 0, numel)
        output_buffer = self.output_torch.narrow(0, 0, numel)
        input_buffer.copy_(inp)
        # TODO(asamani): optimize and only perform all reduce for the current data
        self.all_reduce_jitted(
            self.input_tensor,
            self.output_tensor, 
            self.input_mc_memref, 
            self.output_mc_memref,
            self.num_warps_per_block,
            self.num_ctas,
            self.num_iter_per_thread,
            self.local_rank,
            self.world_size,
            self.cute_tensor_barrier_start,
            self.cute_tensor_barrier_end,
            self.tensor_mc_memerf_barrier_start,
            self.tensor_mc_memerf_barrier_end)
        out.copy_(output_buffer)

    def create_barrier_flags(
        self, 
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device
    ):
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )
        torch_tensor = self.nvshmem_module.nvshmem_malloc(shape, torch_dtype, device)
        torch_tensor.fill_(0)
        tensor_mc = self.nvshmem_module.nvshmem_multicast_ptr(torch_tensor)
        cute_tensor = from_dlpack(torch_tensor)
        cute_tensor.element_type = dtype
        cute_tensor.mark_layout_dynamic()
        cute_tensor_mc = from_dlpack(
            as_tensor(tensor_mc, torch_tensor.shape, torch_tensor.dtype),
        )
        cute_tensor_mc.mark_layout_dynamic()
        return cute_tensor, torch_tensor, cute_tensor_mc


    def create_and_permute_tensor(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        is_mc: bool = False,
        fill_value: Optional[float] = None,
    ):
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )
        torch_tensor = self.nvshmem_module.nvshmem_malloc(shape, torch_dtype, device)
        if fill_value is not None:
            torch_tensor.fill_(fill_value)
        else:
            torch_tensor.copy_(torch.randint(1,16, shape, dtype=torch_dtype, device=device))
        cute_tensor_mc = None
        if is_mc:
            mc_ptr = self.nvshmem_module.nvshmem_multicast_ptr(torch_tensor)
            cute_tensor_mc = from_dlpack(as_tensor(mc_ptr,
                                            torch_tensor.shape,
                                            torch_tensor.dtype),
                                    assumed_align=16)
            cute_tensor_mc = cute_tensor_mc.mark_layout_dynamic()
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        cute_tensor.mark_layout_dynamic()
        # TODO(asamani): if fp8 handle differently
        return cute_tensor, torch_tensor, cute_tensor_mc

    def shutdown(self):
        del self.input_tensor
        del self.output_tensor
        del self.input_mc_memref
        del self.output_mc_memref
        del self.input_torch
        del self.output_torch
        del self.nvshmem_tensor_barrier_start
        del self.nvshmem_tensor_barrier_end
        del self.tensor_mc_memerf_barrier_start
        del self.tensor_mc_memerf_barrier_end
        del self.cute_tensor_barrier_start
        del self.cute_tensor_barrier_end
        torch.distributed.barrier(self.group)
        torch.cuda.synchronize()
        if self.should_init:
            self.nvshmem_module.nvshmem_finalize()