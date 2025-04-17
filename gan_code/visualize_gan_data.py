import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 定义故障类型映射
FAULT_TYPES = {
    'aging': 'rp_features_pv_aging_IV_LD_corrected.csv',
    'normal': 'rp_features_pv_normal_IV_LD_corrected.csv',
    'open': 'rp_features_pv_open_circuit_IV_LD_corrected.csv',
    'shading': 'rp_features_pv_shading_IV_LD_corrected.csv',
    'short': 'rp_features_pv_short_circuit_IV_LD_corrected.csv',
    'aging_open': 'rp_features_aging_open_data_IV_LD_corrected.csv',
    'aging_short': 'rp_features_aging_short_data_IV_LD_corrected.csv'
}

# GAN类型映射
GAN_PREFIXES = {
    'DCGAN': 'dcgan',
    'CTGAN': 'ctgan',
    'infoGAN': 'infogan',
    'LSGAN': 'lsgan',
    'WGAN_GP': 'wgan'
}

def load_data(gan_type, fault_type):
    """加载特定GAN和故障类型的数据"""
    # 加载生成的数据
    if gan_type == 'Weighted_GAN':
        gen_file = os.path.join(
            r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\weighted_gan_results",
            f'weighted_combined_{fault_type}.csv'
        )
    else:
        gen_prefix = GAN_PREFIXES.get(gan_type, gan_type.lower())
        gen_file = os.path.join(
            r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\generated_samples",
            f'{gen_prefix}_{fault_type}.csv'
        )
    
    # 加载真实数据
    real_file = os.path.join(
        r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\LD",
        FAULT_TYPES[fault_type]
    )
    
    # 检查文件是否存在
    if not os.path.exists(gen_file):
        raise FileNotFoundError(f"生成数据文件不存在: {gen_file}")
    if not os.path.exists(real_file):
        raise FileNotFoundError(f"真实数据文件不存在: {real_file}")
        
    # 读取数据
    gen_data = pd.read_csv(gen_file)
    real_data = pd.read_csv(real_file)
    
    return gen_data, real_data

def visualize_data(gen_data, real_data, gan_type, fault_type, save_path=None):
    """可视化生成数据"""
    # 创建简洁的XY图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 只绘制生成数据，不包含真实数据
    ax.scatter(gen_data['RP_Before_MPP'], gen_data['RP_After_MPP'], 
               color='blue', alpha=0.7, edgecolors='none')
    

    
    # 移除额外的元素，保持简洁
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
        
    plt.close(fig)

def main():
    # 直接在代码中设置参数，而不是从命令行获取
    gan_type = 'DCGAN'  # 可选: CTGAN, DCGAN, infoGAN, LSGAN, WGAN_GP, Weighted_GAN
    fault_type = 'aging_open'  # 可选: aging, normal, open, shading, short, aging_open, aging_short
    save_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\output_images\dcgan_aging_open.jpg"  # 可选，不保存则设为None
    
    # 验证输入
    if gan_type not in list(GAN_PREFIXES.keys()) + ['Weighted_GAN']:
        print(f"无效的GAN类型: {gan_type}")
        print(f"可用选项: {list(GAN_PREFIXES.keys()) + ['Weighted_GAN']}")
        return
    
    if fault_type not in FAULT_TYPES:
        print(f"无效的故障类型: {fault_type}")
        print(f"可用选项: {list(FAULT_TYPES.keys())}")
        return
    
    try:
        print(f"正在加载 {gan_type} 模型生成的 {fault_type} 故障类型数据...")
        gen_data, real_data = load_data(gan_type, fault_type)
        
        print("数据加载成功，正在绘图...")
        print(f"生成数据量: {len(gen_data)}")
        
        # 显示数据范围
        print("\n生成数据范围:")
        print(f"RP_Before_MPP: [{gen_data['RP_Before_MPP'].min():.2f}, {gen_data['RP_Before_MPP'].max():.2f}]")
        print(f"RP_After_MPP: [{gen_data['RP_After_MPP'].min():.2f}, {gen_data['RP_After_MPP'].max():.2f}]")
        
        # 创建保存目录（如果需要）
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
        visualize_data(gen_data, real_data, gan_type, fault_type, save_path)
        print("可视化完成！")
        
    except Exception as e:
        print(f"出错: {str(e)}")

if __name__ == "__main__":
    main()
