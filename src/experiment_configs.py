"""
实验配置示例 - 快速切换不同模型变体

使用方法：
1. 复制所需的配置到 train_torch_filter.py 的开头
2. 或直接在 main_kitti.py 中指定参数
"""

# ============================================================================
# 配置1: Baseline CNN (原始AI-IMU-DR)
# ============================================================================
CONFIG_BASELINE = {
    'model_file': 'utils_torch_filter',
    'description': 'Original CNN-based MesNet',
    'params': {
        
    }
}

# ============================================================================
# 配置2: Fixed Window Attention - 小窗口
# ============================================================================
CONFIG_FIXED_SMALL = {
    'model_file': 'utils_torch_filter_winattn',
    'description': 'Fixed Window Attention (window_size=20, ~0.2s)',
    'params': {
        'window_size': 20,
        'num_heads': 4,
    }
}

# ============================================================================
# 配置3: Fixed Window Attention - 标准窗口
# ============================================================================
CONFIG_FIXED_STANDARD = {
    'model_file': 'utils_torch_filter_winattn',
    'description': 'Fixed Window Attention (window_size=50, ~0.5s)',
    'params': {
        'window_size': 50,
        'num_heads': 4,
    }
}

# ============================================================================
# 配置4: Fixed Window Attention - 大窗口
# ============================================================================
CONFIG_FIXED_LARGE = {
    'model_file': 'utils_torch_filter_winattn',
    'description': 'Fixed Window Attention (window_size=100, ~1.0s)',
    'params': {
        'window_size': 100,
        'num_heads': 4,
    }
}

# ============================================================================
# 配置5: Dynamic Window Attention - 保守范围
# ============================================================================
CONFIG_DYNAMIC_CONSERVATIVE = {
    'model_file': 'utils_torch_filter_dynwinattn',
    'description': 'Dynamic Window (10-50 steps, ~0.1-0.5s)',
    'params': {
        'min_window': 10,
        'max_window': 50,
        'num_heads': 4,
        'num_candidates': 10,
        'temperature': 5.0,
    }
}

# ============================================================================
# 配置6: Dynamic Window Attention - 标准范围（推荐）
# ============================================================================
CONFIG_DYNAMIC_STANDARD = {
    'model_file': 'utils_torch_filter_dynwinattn',
    'description': 'Dynamic Window (10-100 steps, ~0.1-1.0s) [RECOMMENDED]',
    'params': {
        'min_window': 10,
        'max_window': 100,
        'num_heads': 4,
        'num_candidates': 10,
        'temperature': 5.0,
    }
}

# ============================================================================
# 配置7: Dynamic Window Attention - 激进范围
# ============================================================================
CONFIG_DYNAMIC_AGGRESSIVE = {
    'model_file': 'utils_torch_filter_dynwinattn',
    'description': 'Dynamic Window (20-200 steps, ~0.2-2.0s)',
    'params': {
        'min_window': 20,
        'max_window': 200,
        'num_heads': 4,
        'num_candidates': 10,
        'temperature': 5.0,
    }
}

# ============================================================================
# 配置8: Dynamic Window - 多注意力头消融
# ============================================================================
CONFIG_DYNAMIC_HEADS_1 = {
    'model_file': 'utils_torch_filter_dynwinattn',
    'description': 'Dynamic Window with 1 head',
    'params': {
        'min_window': 10,
        'max_window': 100,
        'num_heads': 1,
        'num_candidates': 10,
        'temperature': 5.0,
    }
}

CONFIG_DYNAMIC_HEADS_8 = {
    'model_file': 'utils_torch_filter_dynwinattn',
    'description': 'Dynamic Window with 8 heads',
    'params': {
        'min_window': 10,
        'max_window': 100,
        'num_heads': 8,
        'num_candidates': 10,
        'temperature': 5.0,
    }
}


# ============================================================================
# 使用示例
# ============================================================================

def apply_config(config):
    """
    应用配置到训练脚本
    
    Usage:
        from experiment_configs import CONFIG_DYNAMIC_STANDARD, apply_config
        apply_config(CONFIG_DYNAMIC_STANDARD)
    """
    print(f"Applying configuration: {config['description']}")
    print(f"Model file: {config['model_file']}")
    print(f"Parameters: {config['params']}")
    
    # 这里可以添加自动修改import的逻辑
    # 或者返回配置信息供手动修改
    return config


def get_all_configs():
    """获取所有可用配置"""
    configs = {
        'baseline': CONFIG_BASELINE,
        'fixed_small': CONFIG_FIXED_SMALL,
        'fixed_standard': CONFIG_FIXED_STANDARD,
        'fixed_large': CONFIG_FIXED_LARGE,
        'dynamic_conservative': CONFIG_DYNAMIC_CONSERVATIVE,
        'dynamic_standard': CONFIG_DYNAMIC_STANDARD,
        'dynamic_aggressive': CONFIG_DYNAMIC_AGGRESSIVE,
        'dynamic_heads_1': CONFIG_DYNAMIC_HEADS_1,
        'dynamic_heads_8': CONFIG_DYNAMIC_HEADS_8,
    }
    return configs


def print_comparison_table():
    """打印配置对比表"""
    configs = get_all_configs()
    
    print("\n" + "="*80)
    print("模型配置对比表")
    print("="*80)
    print(f"{'配置名称':<30} {'窗口范围':<20} {'注意力头':<10}")
    print("-"*80)
    
    for name, config in configs.items():
        params = config['params']
        
        if 'window_size' in params:
            window_range = f"{params['window_size']} (固定)"
        elif 'min_window' in params:
            window_range = f"{params['min_window']}-{params['max_window']}"
        else:
            window_range = "N/A (CNN)"
        
        num_heads = params.get('num_heads', 'N/A')
        
        print(f"{name:<30} {window_range:<20} {num_heads:<10}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    print_comparison_table()
    
    print("\n推荐的实验顺序：")
    print("1. baseline → 建立性能基线")
    print("2. fixed_standard → 验证窗口注意力的有效性")
    print("3. dynamic_standard → 测试自适应窗口的优势")
    print("4. 消融实验 → 调整窗口范围和注意力头数")
