import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False     # 正确显示负号

FAULT_COLORS = {
    'DOA+OC': '#1f77b4',  # 蓝色
    'DOA+SC': '#ff7f0e',  # 橙色
    'DOA': '#2ca02c',     # 绿色
    'NS': '#d62728',      # 红色
    'OC': '#9467bd',      # 紫色
    'PS': '#8c564b',      # 棕色
    'SC': '#e377c2'       # 粉色
}


def plot_gan_features(gan_type, data_type='generated'):
    """绘制指定GAN类型的数据分布图"""
    # 根据数据类型选择基础文件夹路径
    base_folder = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code"
    if data_type == 'generated':
        features_folder = os.path.join(base_folder, "generated_samples")
        output_name = f'{gan_type}_generated_features_distribution.jpg'
    else:
        features_folder = os.path.join(base_folder, "weighted_gan_results")
        output_name = f'{gan_type}_weighted_features_distribution.jpg'
    
    # 文件名模式映射
    file_patterns = {
        'aging_open': 'DOA+OC',
        'aging_short': 'DOA+SC',
        'aging': 'DOA',
        'normal': 'NS',
        'open': 'OC',
        'shading': 'PS',
        'short': 'SC',
    }
    
    # 获取所有对应GAN类型的CSV文件
    gan_prefix = gan_type.lower()  # 转换为小写进行匹配
    csv_files = [f for f in os.listdir(features_folder) 
                if f.endswith('.csv') and f.startswith(f'{gan_prefix}_')]
    
    print(f"找到 {len(csv_files)} 个 {gan_type} 特征文件...")
    
    # 创建有序数据存储
    ordered_data = {pattern: {'x': [], 'y': [], 'color': None} for pattern in file_patterns.keys()}
    
    # 处理每个文件
    for i, csv_file in enumerate(csv_files):
        try:
            print(f"\n正在处理文件: {csv_file}")
            file_path = os.path.join(features_folder, csv_file)
            
            # 读取数据
            df = pd.read_csv(file_path)
            x_data = df.iloc[:, 0].values  # 第一列作为x轴数据
            y_data = df.iloc[:, 1].values  # 第二列作为y轴数据
            
            # 去除NaN值和异常值
            mask = (~np.isnan(x_data) & ~np.isnan(y_data) & 
                   (np.abs(x_data) < 300) & (np.abs(y_data) < 300))
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            # 将数据存储到对应的故障类型中
            for pattern in file_patterns.keys():
                if pattern in csv_file.lower():
                    ordered_data[pattern]['x'] = x_data
                    ordered_data[pattern]['y'] = y_data
                    ordered_data[pattern]['color'] = None  # 不指定颜色，使用默认色系
                    break
                    
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {str(e)}")
    
    # 按file_patterns的顺序绘制散点图
    plt.figure(figsize=(12, 8))
    all_x_data = []
    all_y_data = []
    
    for pattern, legend_name in file_patterns.items():
        data = ordered_data[pattern]
        if len(data['x']) > 0:
            plt.scatter(data['x'], data['y'],
                       label=legend_name,
                       color=FAULT_COLORS[legend_name],  # 使用预定义的颜色
                       alpha=0.6)
            all_x_data.extend(data['x'])
            all_y_data.extend(data['y'])
    
    if all_x_data and all_y_data:
        # 设置图形属性
        title = 'Generated Data Distribution' if data_type == 'generated' else 'Weighted Data Distribution'
        plt.title(title, fontsize=14)
        
        # 调整坐标轴范围
        x_min, x_max = np.min(all_x_data), np.max(all_x_data)
        y_min, y_max = np.min(all_y_data), np.max(all_y_data)
        
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        plt.xlim(max(-300, x_min - x_margin), min(300, x_max + x_margin))
        plt.ylim(max(-300, y_min - y_margin), min(300, y_max + y_margin))
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        save_path = os.path.join(features_folder, output_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\n分布图已保存到: {save_path}")
        
        # 显示图形
        plt.show()
    else:
        print("\n警告：没有找到有效的数据点，无法生成图形")

if __name__ == "__main__":
    # GAN类型列表（使用小写以匹配文件名）
    gan_types = ['ctgan', 'dcgan', 'wgan', 'lsgan', 'infogan', 'diffusion', 'weighted_combined']
    
    for gan_type in gan_types:
        print(f"\n处理 {gan_type.upper()} 数据...")
        try:
            # 生成原始数据分布图
            print(f"生成 {gan_type.upper()} 原始数据分布图...")
            plot_gan_features(gan_type, 'generated')
            plt.close()
            
            # 生成加权数据分布图
            print(f"生成 {gan_type.upper()} 加权数据分布图...")
            plot_gan_features(gan_type, 'weighted')
            plt.close()
            
        except Exception as e:
            print(f"处理 {gan_type.upper()} 时出错: {str(e)}")