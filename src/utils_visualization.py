"""
窗口注意力可视化工具

用于分析和可视化固定窗口和动态窗口注意力机制的行为
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def visualize_attention_weights(attention_module, input_data, title="Attention Weights"):
    """
    可视化注意力权重
    
    Args:
        attention_module: WindowAttention 或 DynamicWindowAttention 模块
        input_data: 输入数据 (B, T, d_model)
        title: 图表标题
    """
    attention_module.eval()
    
    with torch.no_grad():
        if hasattr(attention_module, 'window_predictor'):
            # Dynamic Window Attention
            output, expected_windows = attention_module(input_data)
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Expected window sizes over time
            ax1 = axes[0]
            batch_idx = 0
            ax1.plot(expected_windows[batch_idx].cpu().numpy(), linewidth=2)
            ax1.set_xlabel('Time Step', fontsize=12)
            ax1.set_ylabel('Window Size', fontsize=12)
            ax1.set_title(f'Expected Window Sizes Over Time (Batch {batch_idx})', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=attention_module.min_window, color='r', linestyle='--', label=f'Min ({attention_module.min_window})')
            ax1.axhline(y=attention_module.max_window, color='g', linestyle='--', label=f'Max ({attention_module.max_window})')
            ax1.legend()
            
            # Plot 2: Window size distribution
            ax2 = axes[1]
            windows_flat = expected_windows.flatten().cpu().numpy()
            ax2.hist(windows_flat, bins=30, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Window Size', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Window Size Distribution', fontsize=14)
            ax2.axvline(x=np.mean(windows_flat), color='r', linestyle='--', label=f'Mean: {np.mean(windows_flat):.1f}')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('dynamic_window_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        else:
            # Fixed Window Attention
            output = attention_module(input_data)
            
            # For fixed window, we can't directly visualize attention weights
            # without modifying the forward pass to return them
            print("Fixed Window Attention - output shape:", output.shape)
            print(f"Window size: {attention_module.window_size}")


def plot_window_comparison(window_sizes, metrics, save_path='window_comparison.png'):
    """
    比较不同窗口大小的性能指标
    
    Args:
        window_sizes: 窗口大小列表
        metrics: 对应的性能指标列表（如ATE、RPE等）
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(window_sizes, metrics, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Window Size (time steps)', fontsize=14)
    ax.set_ylabel('Performance Metric (lower is better)', fontsize=14)
    ax.set_title('Window Size vs Performance', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Find and mark the best window size
    best_idx = np.argmin(metrics)
    ax.plot(window_sizes[best_idx], metrics[best_idx], 'r*', markersize=20, 
            label=f'Best: {window_sizes[best_idx]}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Best window size: {window_sizes[best_idx]} with metric: {metrics[best_idx]:.4f}")


def visualize_dynamic_vs_fixed(dynamic_stats, fixed_stats, save_path='comparison.png'):
    """
    比较动态窗口和固定窗口的性能
    
    Args:
        dynamic_stats: 动态窗口统计信息字典
        fixed_stats: 固定窗口统计信息字典
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Plot 1: Training loss comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in dynamic_stats and 'train_loss' in fixed_stats:
        ax1.plot(dynamic_stats['train_loss'], label='Dynamic Window', linewidth=2)
        ax1.plot(fixed_stats['train_loss'], label='Fixed Window', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss Comparison', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation metric comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if 'val_metric' in dynamic_stats and 'val_metric' in fixed_stats:
        ax2.plot(dynamic_stats['val_metric'], label='Dynamic Window', linewidth=2)
        ax2.plot(fixed_stats['val_metric'], label='Fixed Window', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Metric', fontsize=12)
        ax2.set_title('Validation Metric Comparison', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Window size distribution (dynamic only)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'window_distribution' in dynamic_stats:
        windows = dynamic_stats['window_distribution']
        ax3.hist(windows, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.set_xlabel('Window Size', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Dynamic Window Size Distribution', fontsize=14)
        ax3.axvline(x=np.mean(windows), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(windows):.1f}')
        ax3.legend()
    
    # Plot 4: Performance summary
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['ATE', 'RPE', 'Training Time']
    dynamic_values = [
        dynamic_stats.get('ate', 0),
        dynamic_stats.get('rpe', 0),
        dynamic_stats.get('training_time', 0)
    ]
    fixed_values = [
        fixed_stats.get('ate', 0),
        fixed_stats.get('rpe', 0),
        fixed_stats.get('training_time', 0)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, dynamic_values, width, label='Dynamic', color='steelblue')
    bars2 = ax4.bar(x + width/2, fixed_values, width, label='Fixed', color='coral')
    
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Performance Summary', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_motion_patterns(imu_data, window_sizes, save_path='motion_analysis.png'):
    """
    分析运动模式与窗口大小的关系
    
    Args:
        imu_data: IMU数据 (T, 6) - [omega_x, omega_y, omega_z, acc_x, acc_y, acc_z]
        window_sizes: 预测的窗口大小序列 (T,)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    t = np.arange(len(imu_data))
    
    # Plot 1: Angular velocity magnitude
    ax1 = axes[0]
    omega_mag = np.linalg.norm(imu_data[:, :3], axis=1)
    ax1.plot(t, omega_mag, linewidth=1.5, color='steelblue')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Angular Velocity Magnitude', fontsize=12)
    ax1.set_title('Motion Pattern Analysis', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Acceleration magnitude
    ax2 = axes[1]
    acc_mag = np.linalg.norm(imu_data[:, 3:], axis=1)
    ax2.plot(t, acc_mag, linewidth=1.5, color='coral')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Acceleration Magnitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Window size overlay
    ax3 = axes[2]
    ax3.plot(t, window_sizes, linewidth=2, color='green', label='Window Size')
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Window Size', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute correlation
    motion_intensity = omega_mag + acc_mag
    correlation = np.corrcoef(motion_intensity, window_sizes)[0, 1]
    print(f"Correlation between motion intensity and window size: {correlation:.3f}")
    
    if correlation > 0.3:
        print("[OK] Positive correlation: Larger windows for high-dynamic motion")
    elif correlation < -0.3:
        print("[WARN] Negative correlation: Unexpected pattern")
    else:
        print("- Weak correlation: Window size relatively independent of motion")


def save_analysis_report(results_dict, save_dir='./analysis_reports'):
    """
    保存分析报告
    
    Args:
        results_dict: 包含所有实验结果的字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    report_file = os.path.join(save_dir, 'experiment_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AI-IMU-DR Window Attention Experiment Report\n")
        f.write("="*80 + "\n\n")
        
        for exp_name, results in results_dict.items():
            f.write(f"\nExperiment: {exp_name}\n")
            f.write("-"*80 + "\n")
            
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric_name}: {value:.6f}\n")
                elif isinstance(value, (list, np.ndarray)):
                    f.write(f"  {metric_name}: {len(value)} samples\n")
                    f.write(f"    Mean: {np.mean(value):.6f}, Std: {np.std(value):.6f}\n")
                else:
                    f.write(f"  {metric_name}: {value}\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Report saved to: {report_file}")


if __name__ == '__main__':
    # Example usage
    print("Window Attention Visualization Tools")
    print("="*80)
    print("\nAvailable functions:")
    print("1. visualize_attention_weights() - Visualize attention patterns")
    print("2. plot_window_comparison() - Compare different window sizes")
    print("3. visualize_dynamic_vs_fixed() - Compare dynamic vs fixed windows")
    print("4. analyze_motion_patterns() - Analyze motion-window relationship")
    print("5. save_analysis_report() - Save experiment report")
    print("\nUsage example:")
    print("  from utils_visualization import *")
    print("  visualize_attention_weights(model.mes_net.dynamic_attention, test_data)")
