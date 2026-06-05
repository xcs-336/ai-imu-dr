#!/usr/bin/env python3
"""
AI-IMU-DR Torch 训练代码诊断脚本
用于检查所有必需的函数、类和导入是否正确定义
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_import(module_name, import_list):
    """检查模块导入"""
    try:
        module = __import__(module_name, fromlist=import_list)
        print(f"✅ {module_name:<40} 导入成功")
        
        # 检查具体的导入项
        for item in import_list:
            if hasattr(module, item):
                print(f"   [OK] {item}")
            else:
                print(f"   [FAIL] {item} - 未找到")
        return True
    except Exception as e:
        print(f"❌ {module_name:<40} 导入失败")
        print(f"   错误: {e}")
        return False

def check_class_methods(module, class_name, required_methods):
    """检查类的方法是否都存在"""
    try:
        cls = getattr(module, class_name)
        print(f"\n检查类: {class_name}")
        
        missing = []
        for method in required_methods:
            if hasattr(cls, method):
                print(f"   [OK] {method}")
            else:
                print(f"   [FAIL] {method} - 缺失")
                missing.append(method)
        
        if missing:
            print(f"   ⚠️  缺失 {len(missing)} 个方法: {', '.join(missing)}")
            return False
        else:
            print(f"   ✅ 所有 {len(required_methods)} 个方法都存在")
            return True
    except Exception as e:
        print(f"   ❌ 检查失败: {e}")
        return False

def main():
    print_section("AI-IMU-DR Torch 训练代码诊断工具")
    
    print("\n📋 检查 Python 环境...")
    print(f"   Python 版本: {sys.version}")
    
    # 检查 PyTorch
    try:
        import torch
        print(f"   PyTorch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   ❌ PyTorch 未安装")
        return
    
    # 检查其他依赖
    print("\n📋 检查依赖包...")
    dependencies = ['numpy', 'matplotlib', 'scipy', 'navpy', 'termcolor']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - 未安装")
    
    # 检查核心模块导入
    print_section("检查核心模块导入")
    
    modules_to_check = [
        ('utils_numpy_filter', ['NUMPYIEKF']),
        ('utils', ['prepare_data', 'create_folder', 'umeyama_alignment']),
        ('dataset', ['BaseDataset']),
        ('experiment_configs', ['get_all_configs']),
    ]
    
    for module_name, imports in modules_to_check:
        check_import(module_name, imports)
    
    # 检查三个模型文件
    print_section("检查模型文件")
    
    model_files = [
        ('utils_torch_filter', 'Baseline CNN'),
        ('utils_torch_filter_winattn', 'Fixed Window Attention'),
        ('utils_torch_filter_dynwinattn', 'Dynamic Window Attention'),
    ]
    
    required_methods = [
        'set_Q',
        'forward_nets',
        'run',
        'load',
        'get_normalize_u',
        'set_param_attr',
        'normalize_u',
    ]
    
    all_models_ok = True
    for module_name, description in model_files:
        print(f"\n{description} ({module_name}):")
        if check_import(module_name, ['TORCHIEKF']):
            try:
                module = __import__(module_name, fromlist=['TORCHIEKF'])
                if not check_class_methods(module, 'TORCHIEKF', required_methods):
                    all_models_ok = False
            except Exception as e:
                print(f"   ❌ 检查失败: {e}")
                all_models_ok = False
    
    # 检查训练模块
    print_section("检查训练模块")
    
    if check_import('train_torch_filter', ['train_filter', 'load_model_for_config', 'parse_args']):
        try:
            import train_torch_filter
            print("\n检查训练模块函数:")
            train_functions = [
                'train_filter',
                'prepare_filter',
                'prepare_loss_data',
                'train_loop',
                'mini_batch_step',
                'save_iekf',
                'set_optimizer',
                'precompute_lost',
            ]
            for func in train_functions:
                if hasattr(train_torch_filter, func):
                    print(f"   [OK] {func}")
                else:
                    print(f"   [FAIL] {func} - 缺失")
        except Exception as e:
            print(f"   ❌ 检查失败: {e}")
    
    # 检查主程序
    print_section("检查主程序")
    
    if check_import('main_kitti', ['KITTIDataset', 'KITTIParameters', 'launch']):
        try:
            from main_kitti import KITTIDataset, KITTIArgs
            
            print("\n检查 KITTIArgs 配置:")
            args = KITTIArgs()
            required_attrs = [
                'path_data_base',
                'path_data_save',
                'path_results',
                'path_temp',
                'epochs',
                'seq_dim',
                'config',
                'train_filter',
                'test_filter',
                'results_filter',
            ]
            for attr in required_attrs:
                if hasattr(args, attr):
                    value = getattr(args, attr)
                    print(f"   [OK] {attr}: {value}")
                else:
                    print(f"   [FAIL] {attr} - 缺失")
        except Exception as e:
            print(f"   ❌ 检查失败: {e}")
    
    # 检查 utils_plot
    print_section("检查可视化工具")
    
    if check_import('utils_plot', ['results_filter']):
        print("   ✅ results_filter 函数可用")
    
    # 总结
    print_section("诊断总结")
    
    print("\n✅ 所有基本检查完成！")
    print("\n建议的下一步:")
    print("1. 如果所有检查都通过，可以尝试运行:")
    print("   python src/main_kitti.py --config baseline --train --epochs 5")
    print("\n2. 如果有缺失的依赖，请安装:")
    print("   pip install matplotlib numpy termcolor scipy navpy torch")
    print("\n3. 如果需要下载数据，请参考 README_ZH.md")
    print("\n4. 更多问题排查请参考 TROUBLESHOOTING_TORCH.md")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
