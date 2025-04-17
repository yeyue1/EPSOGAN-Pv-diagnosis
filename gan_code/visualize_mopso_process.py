import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义故障类型
FAULT_TYPES = [
    'aging', 'normal', 'open', 'shading', 
    'short', 'aging_open', 'aging_short'
]

# 定义GAN类型
GAN_TYPES = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']

def load_mopso_results(fault_type):
    """加载指定故障类型的MOPSO优化结果"""
    file_path = f"C:\\Users\\yeyue\\Desktop\\实验室工作用\\论文2\\Paper_Code\\gan_code\\mopso_results\\mopso_results_{fault_type}.csv"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"无法找到文件: {file_path}")
        return None
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    return df

def load_summary_results():
    """加载优化总结结果"""
    file_path = f"C:\\Users\\yeyue\\Desktop\\实验室工作用\\论文2\\Paper_Code\\gan_code\\mopso_results\\optimization_summary.csv"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"无法找到总结文件: {file_path}")
        return None
    
    # 读取CSV文件
    df = pd.read_csv(file_path, index_col=0)
    return df

def visualize_pareto_front_2d(fault_type, save_dir=None):
    """生成2D帕累托前沿可视化"""
    # 加载数据
    df = load_mopso_results(fault_type)
    if df is None:
        return
        
    # 确保有足够的数据点
    if len(df) < 2:
        print(f"{fault_type}的数据点太少，无法绘制帕累托前沿")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制所有解 - MMD vs Coverage
    sc = ax.scatter(df['MMD'], df['Coverage'], 
                   c=df['Wasserstein'], cmap='viridis', 
                   s=50, alpha=0.7, edgecolors='w')
    
    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Wasserstein距离', fontsize=12)
    
    # 查找总结文件中的最优解
    summary_df = load_summary_results()
    if summary_df is not None and fault_type in summary_df.index:
        best_mmd = summary_df.loc[fault_type, 'MMD']
        best_coverage = summary_df.loc[fault_type, 'Coverage']
        
        # 标记最优解
        ax.scatter([best_mmd], [best_coverage], 
                  color='red', s=150, marker='*', 
                  label='最优解', edgecolors='k', linewidths=1.5)
    
    # 设置轴标签和标题
    ax.set_xlabel('MMD (越小越好)', fontsize=14)
    ax.set_ylabel('Coverage (越大越好)', fontsize=14)
    ax.set_title(f'{fault_type}故障类型的帕累托前沿', fontsize=16)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 保存图表
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'pareto_front_{fault_type}.png'), dpi=300, bbox_inches='tight')
        print(f"已保存{fault_type}的帕累托前沿图")
    else:
        plt.show()
    
    plt.close(fig)

def visualize_weights_radar(fault_type, save_dir=None):
    """创建权重分布的雷达图"""
    # 加载优化总结结果
    summary_df = load_summary_results()
    if summary_df is None or fault_type not in summary_df.index:
        print(f"无法找到{fault_type}的优化总结结果")
        return
    
    # 提取权重
    weights = [summary_df.loc[fault_type, f'Weight_{gan}'] for gan in GAN_TYPES]
    
    # 创建雷达图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(GAN_TYPES), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    weights += weights[:1]  # 闭合数据
    
    # 绘制雷达图
    ax.plot(angles, weights, 'o-', linewidth=2)
    ax.fill(angles, weights, alpha=0.25)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(GAN_TYPES)
    
    # 设置标题
    ax.set_title(f'{fault_type}故障类型的最优权重分布', fontsize=16)
    
    # 保存图表
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'weights_radar_{fault_type}.png'), dpi=300, bbox_inches='tight')
        print(f"已保存{fault_type}的权重雷达图")
    else:
        plt.show()
    
    plt.close(fig)

def visualize_weights_pie(fault_type, save_dir=None):
    """创建权重分布的饼图"""
    # 加载优化总结结果
    summary_df = load_summary_results()
    if summary_df is None or fault_type not in summary_df.index:
        print(f"无法找到{fault_type}的优化总结结果")
        return
    
    # 提取权重
    weights = [summary_df.loc[fault_type, f'Weight_{gan}'] for gan in GAN_TYPES]
    
    # 过滤掉权重为0的GAN
    filtered_weights = []
    filtered_labels = []
    for i, w in enumerate(weights):
        if w > 0.001:  # 忽略非常小的权重
            filtered_weights.append(w)
            filtered_labels.append(GAN_TYPES[i])
    
    # 创建饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(filtered_weights, labels=filtered_labels, 
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 14})
    
    # 设置标题
    ax.set_title(f'{fault_type}故障类型的最优GAN权重分布', fontsize=16)
    
    # 添加图例
    ax.legend(wedges, filtered_labels, title="GAN类型", 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # 保存图表
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'weights_pie_{fault_type}.png'), dpi=300, bbox_inches='tight')
        print(f"已保存{fault_type}的权重饼图")
    else:
        plt.show()
    
    plt.close(fig)

def visualize_objective_scores(fault_type, save_dir=None):
    """可视化目标函数值的比较"""
    # 加载优化总结结果
    summary_df = load_summary_results()
    if summary_df is None or fault_type not in summary_df.index:
        print(f"无法找到{fault_type}的优化总结结果")
        return
    
    # 提取评估指标数据
    metrics = ['MMD', 'Coverage', 'Wasserstein', 'ModeScore']
    values = [summary_df.loc[fault_type, m] for m in metrics]
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置条形图颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 绘制条形图
    bars = ax.bar(metrics, values, color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3点垂直偏移
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # 设置标题和轴标签
    ax.set_title(f'{fault_type}故障类型的最优解评估指标', fontsize=16)
    ax.set_xlabel('评估指标', fontsize=14)
    ax.set_ylabel('值', fontsize=14)
    
    # 设置Y轴从0开始
    ax.set_ylim(bottom=0)
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'objective_scores_{fault_type}.png'), dpi=300, bbox_inches='tight')
        print(f"已保存{fault_type}的评估指标图")
    else:
        plt.show()
    
    plt.close(fig)

def visualize_optimization_process(fault_type, save_dir=None):
    """综合可视化优化过程"""
    print(f"\n正在为{fault_type}故障类型创建可视化...")
    
    # 创建帕累托前沿可视化
    visualize_pareto_front_2d(fault_type, save_dir)
    
    # 创建权重分布可视化
    visualize_weights_radar(fault_type, save_dir)
    visualize_weights_pie(fault_type, save_dir)
    
    # 创建评估指标可视化
    visualize_objective_scores(fault_type, save_dir)

def main():
    # 设置可视化保存目录
    save_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_visualization"
    
    # 检查目录是否存在，若不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建保存目录: {save_dir}")
    
    # 处理所有故障类型
    for fault_type in FAULT_TYPES:
        visualize_optimization_process(fault_type, save_dir)
    
    print("\n所有可视化已完成!")

if __name__ == "__main__":
    main()
