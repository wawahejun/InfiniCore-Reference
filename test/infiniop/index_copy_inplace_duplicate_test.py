#!/usr/bin/env python3

import torch
import ctypes
from ctypes import c_uint64, c_int32, c_int
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    create_handle,
    destroy_handle,
)
from enum import Enum, auto
import numpy as np

# ==============================================================================
#  重复索引测试配置
# ==============================================================================

# 专门测试重复索引的测试用例
_DUPLICATE_INDEX_TEST_CASES_ = [
    # (target_shape, source_shape, index_shape, dim, duplicate_indices)
    # 小规模测试用例
    ((5, 3), (3, 3), (3,), 0, [0, 2, 0]),  # 简单重复
    ((4, 4), (4, 4), (4,), 0, [1, 1, 2, 3]),  # 部分重复
    ((6, 2), (4, 2), (4,), 0, [0, 1, 0, 1]),  # 完全重复模式
    
    # 中等规模测试用例
    ((10, 8), (6, 8), (6,), 0, [2, 5, 2, 7, 5, 9]),  # 多个重复
    ((8, 10), (8, 5), (5,), 1, [3, 7, 3, 1, 7]),  # dim=1的重复
    
    # 大规模测试用例
    ((64, 32), (32, 32), (32,), 0, None),  # 使用生成函数创建重复索引
    ((32, 64), (32, 32), (32,), 1, None),  # dim=1的大规模重复
    
    # 3D张量测试用例
    ((8, 6, 4), (4, 6, 4), (4,), 0, [1, 3, 1, 5]),  # 3D重复
    ((6, 8, 4), (6, 4, 4), (4,), 1, [2, 5, 2, 7]),  # 3D dim=1重复

    ((64, 64, 32), (32, 64, 32), (32,), 0, None),  # 3D大尺寸，使用生成函数创建重复索引
    ((32, 128, 16), (32, 64, 16), (64,), 1, None),  # 3D大尺寸，dim=1，使用生成函数创建重复索引
]

# 测试的数据类型（重点测试浮点类型，因为重复索引对浮点数影响最明显）
_DUPLICATE_TEST_DTYPES = [
    InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64, InfiniDtype.BF16,
    InfiniDtype.I32, InfiniDtype.I64
]

# 容差映射
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def index_copy_inplace_torch(target, source, index, dim):
    """PyTorch参考实现"""
    target.index_copy_(dim, index, source)
    return target


def create_duplicate_index(target_shape, source_shape, dim, duplicate_indices=None, dtype=torch.int64):
    """创建包含重复索引的索引张量"""
    target_size = target_shape[dim]
    source_size = source_shape[dim]
    
    if duplicate_indices is not None:
        # 使用预定义的重复索引
        indices = torch.tensor(duplicate_indices, dtype=dtype)
        assert len(indices) == source_size, f"索引长度 {len(indices)} 不匹配源张量大小 {source_size}"
        assert all(0 <= idx < target_size for idx in indices), f"索引超出范围 [0, {target_size})"
    else:
        # 生成确定性的重复索引模式
        # 使用固定的种子确保结果可重现
        torch.manual_seed(42)
        
        # 创建一个确定性的重复模式
        # 策略：前一半索引使用顺序索引，后一半重复前一半的部分索引
        indices = torch.zeros(source_size, dtype=dtype)
        
        # 确保有足够的目标位置
        available_targets = min(target_size, source_size)
        
        # 前一半使用顺序索引
        first_half = source_size // 2
        for i in range(first_half):
            indices[i] = i % available_targets
        
        # 后一半重复前一半的索引，创建重复模式
        for i in range(first_half, source_size):
            # 重复前面的索引，确保有重复
            repeat_idx = (i - first_half) % first_half
            indices[i] = indices[repeat_idx]
    
    return indices


def test_duplicate_index(
    handle,
    device,
    target_shape,
    source_shape,
    index_shape,
    dim,
    duplicate_indices=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    """测试重复索引的index_copy_inplace操作"""
    
    # 创建测试张量
    if dtype in [InfiniDtype.I32, InfiniDtype.I64]:
        target = TestTensor(target_shape, None, dtype, device, mode="random")
        source = TestTensor(source_shape, None, dtype, device, mode="random")
    else:
        # 浮点类型使用更明显的数值来观察重复索引的影响
        target = TestTensor(target_shape, None, dtype, device)
        source = TestTensor(source_shape, None, dtype, device)
        # 为source张量设置更明显的数值模式
        source_data = source.torch_tensor()
        for i in range(source_shape[dim]):
            if dim == 0:
                source_data[i] = (i + 1) * 10.0  # 10, 20, 30, ...
            elif dim == 1:
                source_data[:, i] = (i + 1) * 10.0
            elif dim == 2:
                source_data[:, :, i] = (i + 1) * 10.0
    
    # 创建包含重复的索引
    index_tensor = create_duplicate_index(target_shape, source_shape, dim, duplicate_indices)
    index = TestTensor(index_shape, None, InfiniDtype.I64, device)
    index.torch_tensor().copy_(index_tensor)
    
    # 创建目标张量的副本用于PyTorch参考计算
    target_torch = target.torch_tensor().clone()
    
    print(
        f"测试重复索引 IndexCopyInplace on {InfiniDeviceNames[device]} with target_shape:{target_shape} "
        f"source_shape:{source_shape} index_shape:{index_shape} dim:{dim} dtype:{InfiniDtypeNames[dtype]}"
    )
    print(f"重复索引: {index_tensor.tolist()}")
    
    # 检查是否确实有重复索引
    unique_indices = torch.unique(index_tensor)
    has_duplicates = len(unique_indices) < len(index_tensor)
    print(f"包含重复索引: {has_duplicates}")
    
    if not has_duplicates:
        print("警告: 当前索引不包含重复值，跳过此测试")
        return
    
    # PyTorch参考计算
    index_copy_inplace_torch(target_torch, source.torch_tensor(), index.torch_tensor(), dim)
    
    if sync is not None:
        sync()
    
    # 创建InfiniOp描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateIndexCopyInplaceDescriptor(
            handle,
            ctypes.byref(descriptor),
            target.descriptor,
            source.descriptor,
            c_int32(dim),
            index.descriptor,
        )
    )
    
    # 获取工作空间大小
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetIndexCopyInplaceWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, target.device)
    
    def lib_index_copy_inplace():
        check_error(
            LIBINFINIOP.infiniopIndexCopyInplace(
                descriptor,
                workspace.data(),
                workspace.size(),
                target.data(),
                source.data(),
                index.data(),
                None,
            )
        )
    
    # 执行InfiniOp操作
    lib_index_copy_inplace()
    
    # 检查正确性
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    if DEBUG:
        print(f"PyTorch结果:\n{target_torch}")
        print(f"InfiniOp结果:\n{target.actual_tensor()}")
        debug(target.actual_tensor(), target_torch, atol=atol, rtol=rtol)
    
    # 对于重复索引，我们期望InfiniOp的行为与PyTorch在相同设备上的行为一致
    try:
        assert torch.allclose(target.actual_tensor(), target_torch, atol=atol, rtol=rtol)
        print("✓ 重复索引测试通过 - InfiniOp与PyTorch行为一致")
    except AssertionError:
        print("✗ 重复索引测试失败 - InfiniOp与PyTorch行为不一致")
        if DEBUG:
            diff = torch.abs(target.actual_tensor() - target_torch)
            max_diff = torch.max(diff)
            print(f"最大差异: {max_diff}")
            print(f"差异位置: {torch.where(diff > atol)}")
        raise
    
    # 性能分析
    if PROFILE:
        target_prof = target.torch_tensor().clone()
        profile_operation(
            "PyTorch",
            lambda: index_copy_inplace_torch(target_prof, source.torch_tensor(), index.torch_tensor(), dim),
            device, NUM_PRERUN, NUM_ITERATIONS
        )
        profile_operation("InfiniOp", lambda: lib_index_copy_inplace(), device, NUM_PRERUN, NUM_ITERATIONS)
    
    check_error(LIBINFINIOP.infiniopDestroyIndexCopyInplaceDescriptor(descriptor))


def run_duplicate_index_tests():
    """运行所有重复索引测试"""
    args = get_args()
    
    global DEBUG, PROFILE, NUM_PRERUN, NUM_ITERATIONS
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    print("\n" + "="*80)
    print("开始重复索引测试 - 验证InfiniOp与PyTorch的行为一致性")
    print("="*80)
    
    devices = get_test_devices(args)
    
    for device in devices:
        print(f"\n测试设备: {InfiniDeviceNames[device]}")
        print("-" * 50)
        
        # 设置设备并创建句柄
        LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0))
        handle = create_handle()
        
        test_count = 0
        passed_count = 0
        
        for test_case in _DUPLICATE_INDEX_TEST_CASES_:
            target_shape, source_shape, index_shape, dim = test_case[:4]
            duplicate_indices = test_case[4] if len(test_case) > 4 else None
            
            for dtype in _DUPLICATE_TEST_DTYPES:
                test_count += 1
                try:
                    test_duplicate_index(
                        handle, device, target_shape, source_shape, 
                        index_shape, dim, duplicate_indices, dtype
                    )
                    passed_count += 1
                except Exception as e:
                    print(f"✗ 测试失败: {e}")
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
        
        print(f"\n设备 {InfiniDeviceNames[device]} 测试结果: {passed_count}/{test_count} 通过")
        
        # 销毁句柄
        destroy_handle(handle)
    
    print("\n" + "="*80)
    print("重复索引测试完成")
    print("="*80)


if __name__ == "__main__":
    run_duplicate_index_tests()
    print("\033[92m重复索引测试全部完成!\033[0m")