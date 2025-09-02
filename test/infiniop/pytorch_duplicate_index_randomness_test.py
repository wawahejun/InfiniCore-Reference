#!/usr/bin/env python3

import torch
import numpy as np
import argparse
from collections import defaultdict

def create_duplicate_index_deterministic(source_size, target_size):
    """创建确定性的重复索引模式（与原测试文件相同）"""
    indices = torch.zeros(source_size, dtype=torch.long)
    first_half = source_size // 2
    
    # 前一半使用顺序索引
    for i in range(first_half):
        indices[i] = i % min(target_size, source_size)
    
    # 后一半重复前一半的索引
    for i in range(first_half, source_size):
        repeat_idx = (i - first_half) % first_half
        indices[i] = indices[repeat_idx]
    
    return indices

def test_pytorch_randomness(target_shape, source_shape, dim, num_runs=100, device='cpu'):
    """测试PyTorch在重复索引时的行为随机性"""
    print(f"\n=== 测试配置 ===")
    print(f"target_shape: {target_shape}")
    print(f"source_shape: {source_shape}")
    print(f"dim: {dim}")
    print(f"device: {device}")
    print(f"运行次数: {num_runs}")
    
    # 创建重复索引
    index = create_duplicate_index_deterministic(source_shape[dim], target_shape[dim])
    print(f"\n重复索引: {index[:10].tolist()}...{index[-10:].tolist()}")
    
    # 找到重复的索引位置
    duplicate_positions = {}
    for i, idx in enumerate(index):
        if idx.item() not in duplicate_positions:
            duplicate_positions[idx.item()] = []
        duplicate_positions[idx.item()].append(i)
    
    # 只关注真正有重复的索引
    truly_duplicate = {k: v for k, v in duplicate_positions.items() if len(v) > 1}
    print(f"\n重复索引分析:")
    for idx, positions in list(truly_duplicate.items())[:5]:  # 只显示前5个
        print(f"  索引 {idx} 出现在位置: {positions}")
    
    # 多次运行测试
    results = defaultdict(list)
    
    for run in range(num_runs):
        # 每次创建新的张量
        if device == 'cuda' and torch.cuda.is_available():
            target = torch.zeros(*target_shape, dtype=torch.float32, device='cuda')
            source = torch.ones(*source_shape, dtype=torch.float32, device='cuda')
            index_gpu = index.cuda()
        else:
            target = torch.zeros(*target_shape, dtype=torch.float32)
            source = torch.ones(*source_shape, dtype=torch.float32)
            index_gpu = index
        
        # 设置source的值模式
        for i in range(source_shape[dim]):
            if dim == 0:
                source[i] = (i + 1) * 10.0
            elif dim == 1:
                source[:, i] = (i + 1) * 10.0
            elif dim == 2:
                source[:, :, i] = (i + 1) * 10.0
        
        # 执行index_copy_操作
        target.index_copy_(dim, index_gpu, source)
        
        # 记录重复索引位置的结果
        for idx in truly_duplicate.keys():
            if dim == 0:
                if len(target_shape) == 3:
                    result = target[idx, 0, 0].item()
                elif len(target_shape) == 2:
                    result = target[idx, 0].item()
                else:
                    result = target[idx].item()
            elif dim == 1:
                if len(target_shape) == 3:
                    result = target[0, idx, 0].item()
                elif len(target_shape) == 2:
                    result = target[0, idx].item()
                else:
                    result = target[idx].item()
            elif dim == 2:
                result = target[0, 0, idx].item()
            
            results[idx].append(result)
        
        if run < 10 or run % (num_runs // 10) == 0:  # 显示前10次和每10%的进度
            sample_idx = list(truly_duplicate.keys())[0]
            sample_result = results[sample_idx][-1]
            print(f"运行 {run+1:3d}: 索引{sample_idx}的结果 = {sample_result}")
    
    # 分析结果
    print(f"\n=== 结果分析 ===")
    
    for idx in list(truly_duplicate.keys())[:3]:  # 分析前3个重复索引
        positions = truly_duplicate[idx]
        values = results[idx]
        unique_values = set(values)
        
        print(f"\n索引 {idx} (出现在位置 {positions}):")
        print(f"  不同结果数量: {len(unique_values)}")
        print(f"  结果值: {sorted(unique_values)}")
        
        # 计算期望值
        first_write_value = (positions[0] + 1) * 10.0
        last_write_value = (positions[-1] + 1) * 10.0
        
        print(f"  期望值分析:")
        print(f"    如果第一个值胜出: {first_write_value}")
        print(f"    如果最后值胜出: {last_write_value}")
        
        # 统计结果分布
        value_counts = {}
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1
        
        print(f"  结果分布:")
        for value, count in sorted(value_counts.items()):
            percentage = count / len(values) * 100
            print(f"    {value}: {count}/{len(values)} ({percentage:.1f}%)")
        
        # 判断行为类型
        if len(unique_values) == 1:
            if list(unique_values)[0] == first_write_value:
                print(f"  行为: 确定性 - 第一个值胜出")
            elif list(unique_values)[0] == last_write_value:
                print(f"  行为: 确定性 - 最后值胜出")
            else:
                print(f"  行为: 确定性 - 未知策略")
        else:
            print(f"  行为: 随机性 - 结果不确定")

def main():
    parser = argparse.ArgumentParser(description='测试PyTorch重复索引行为的随机性')
    parser.add_argument('--runs', type=int, default=100, help='测试运行次数')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='测试设备')
    parser.add_argument('--all-cases', action='store_true', help='测试所有案例')
    
    args = parser.parse_args()
    
    # 测试案例（来自原测试文件的大尺寸案例）
    test_cases = [
        # 原测试文件中失败的大尺寸案例
        ((64, 64, 32), (32, 64, 32), 0),  # 3D大尺寸，dim=0
        ((32, 128, 16), (32, 64, 16), 1),  # 3D大尺寸，dim=1
    ]
    
    print("="*80)
    print("PyTorch 重复索引行为随机性测试")
    print("="*80)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU测试")
        args.device = 'cpu'
    
    if args.all_cases:
        cases_to_test = test_cases
    else:
        # 默认只测试第一个案例
        cases_to_test = test_cases[:1]
    
    for i, (target_shape, source_shape, dim) in enumerate(cases_to_test):
        print(f"\n{'='*60}")
        print(f"测试案例 {i+1}/{len(cases_to_test)}")
        print(f"{'='*60}")
        
        test_pytorch_randomness(
            target_shape=target_shape,
            source_shape=source_shape, 
            dim=dim,
            num_runs=args.runs,
            device=args.device
        )
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()