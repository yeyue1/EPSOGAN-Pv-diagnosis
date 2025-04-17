import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
from torch.serialization import add_safe_globals
from pandas.core.series import Series

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入正确的Generator类
from WGAN_GP import Generator as WGAN_Generator
from CTGAN import Generator as CTGAN_Generator
from DCGAN import Generator as DCGAN_Generator
from LSGAN import Generator as LSGAN_Generator
from infoGAN import Generator as InfoGAN_Generator

# 添加Series到安全全局变量列表
add_safe_globals([Series])

# 在导入语句后添加以下代码
FAULT_TYPES = {
    'aging': 'rp_features_pv_aging_IV_LD_corrected.csv',
    'normal': 'rp_features_pv_normal_IV_LD_corrected.csv',
    'open': 'rp_features_pv_open_circuit_IV_LD_corrected.csv',
    'shading': 'rp_features_pv_shading_IV_LD_corrected.csv',
    'short': 'rp_features_pv_short_circuit_IV_LD_corrected.csv',
    'aging_open': 'rp_features_aging_open_data_IV_LD_corrected.csv',
    'aging_short': 'rp_features_aging_short_data_IV_LD_corrected.csv'
}


def load_gan_generator(generator_name, model_path, device, latent_dim=100, num_classes=7):
    """加载特定的生成器模型"""
    feature_dim = 2
    
    if generator_name == 'WGAN_GP':
        generator = WGAN_Generator(latent_dim, feature_dim, num_classes).to(device)  # 添加num_classes参数
    elif generator_name == 'CTGAN':
        generator = CTGAN_Generator(latent_dim, feature_dim, num_classes).to(device)
    elif generator_name == 'DCGAN':
        generator = DCGAN_Generator(latent_dim, feature_dim, num_classes).to(device)
    elif generator_name == 'LSGAN':
        generator = LSGAN_Generator(latent_dim, feature_dim, num_classes).to(device)
    elif generator_name == 'infoGAN':
        generator = InfoGAN_Generator(62, num_classes, 2, feature_dim).to(device)
    else:
        raise ValueError(f"不支持的生成器类型: {generator_name}")
    
    print(f"正在加载模型: {model_path}")
    save_dict = torch.load(model_path, weights_only=False)
    generator.load_state_dict(save_dict['generator_state_dict'])
    generator.eval()
    
    return generator, save_dict

def generate_samples(generator, gan_type, n_samples, device, latent_dim=100, num_classes=7):
    """根据不同GAN类型生成样本"""
    with torch.no_grad():
        if gan_type == 'WGAN_GP':
            z = torch.randn(n_samples, latent_dim).to(device)
            labels = torch.LongTensor([1] * n_samples).to(device)  # 添加标签输入
            samples = generator(z, labels)  # 修改调用方式
        elif gan_type == 'infoGAN':
            z = torch.randn(n_samples, 62).to(device)
            categorical = torch.zeros(n_samples, num_classes).to(device)
            categorical[:, 1] = 1  # 使用normal类型
            continuous = torch.randn(n_samples, 2).to(device)
            samples = generator(z, categorical, continuous)
        else:
            z = torch.randn(n_samples, latent_dim).to(device)
            labels = torch.LongTensor([1] * n_samples).to(device)
            samples = generator(z, labels)
            
    return samples.cpu().numpy()

def denormalize_data(data, min_vals, max_vals, generator_name):
    """根据不同GAN模型使用相应的反归一化方法"""
    data = np.array(data)
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    
    if generator_name.upper() in ['CTGAN']:
        # CTGAN使用[-1, 1]范围
        normalized = (data + 1) / 2
    else:
        # DCGAN和其他GAN使用[-0.9, 0.9]范围
        normalized = (data + 0.9) / 1.8
    
    # 检查是否有越界值
    normalized = np.clip(normalized, 0, 1)
    
    # 从[0, 1]转回原始范围
    denormalized = normalized * (max_vals - min_vals) + min_vals
    
    return denormalized

def generate_weighted_data_by_fault(weights_file, fault_type, model_base_path, output_dir, n_samples=1000):
    """根据特定故障类型的权重生成数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n处理故障类型: {fault_type}")
    print(f"使用设备: {device}")
    
    # 读取权重数据
    weights_df = pd.read_csv(weights_file)
    weights = {
        'CTGAN': weights_df['Weight_CTGAN'].iloc[0],
        'DCGAN': weights_df['Weight_DCGAN'].iloc[0],
        'infoGAN': weights_df['Weight_infoGAN'].iloc[0],
        'LSGAN': weights_df['Weight_LSGAN'].iloc[0],
        'WGAN_GP': weights_df['Weight_WGAN_GP'].iloc[0]
    }
    
    # 存储合并数据
    combined_data = None
    
    # 处理每个GAN类型
    for gan_type, weight in weights.items():
        if weight > 0:
            print(f"\n处理 {gan_type}, 权重: {weight:.4f}")
            
            # 构建模型文件名
            if gan_type == 'WGAN_GP':
                model_name = f"wgan_gp_generator_{fault_type}.pth"
            else:
                model_name = f"{gan_type.lower()}_generator_{fault_type}.pth"
                
            model_path = os.path.join(model_base_path, model_name)
            
            try:
                # 加载生成器并生成样本
                generator, save_dict = load_gan_generator(gan_type, model_path, device)
                samples = generate_samples(generator, gan_type, n_samples, device)
                
                # 反归一化
                samples = denormalize_data(
                    samples,
                    save_dict['min_vals'],
                    save_dict['max_vals'],
                    gan_type  # 传入生成器类型
                )
                
                # 应用权重
                weighted_samples = samples * weight
                
                # 累加到合并数据中
                if combined_data is None:
                    combined_data = weighted_samples
                else:
                    combined_data += weighted_samples
                    
                print(f"成功处理 {gan_type}")
                
            except Exception as e:
                print(f"处理模型 {model_path} 时出错: {str(e)}")
                continue
    
    # 保存合并后的数据
    if combined_data is not None:
        # 直接保存到输出目录
        output_file = os.path.join(output_dir, f'weighted_combined_{fault_type}.csv')
        df = pd.DataFrame(combined_data, columns=['RP_Before_MPP', 'RP_After_MPP'])
        df.to_csv(output_file, index=False)
        print(f"\n已保存数据到: {output_file}")
        print(f"数据范围:")
        print(f"RP_Before_MPP: [{combined_data[:, 0].min():.4f}, {combined_data[:, 0].max():.4f}]")
        # 修复这行的语法错误
        print(f"RP_After_MPP: [{combined_data[:, 1].min():.4f}, {combined_data[:, 1].max():.4f}]")
    
    # 保存权重信息
    weights_df_local = pd.DataFrame({
        'GAN_Type': list(weights.keys()),
        'Weight': list(weights.values())
    })
    # 直接保存到输出目录
    weights_file = os.path.join(output_dir, f'weights_{fault_type}.csv')
    weights_df_local.to_csv(weights_file, index=False)
    
    print("\n权重信息:")
    print(weights_df_local)

def main():
    # 设置路径
    model_base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\gan_model"
    summary_file = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_results\optimization_summary.csv"
    output_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\weighted_gan_results"
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取最优权重
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"找不到优化总结文件: {summary_file}")
    
    summary_df = pd.read_csv(summary_file, index_col=0)
    
    print("开始生成加权数据...")
    
    # 处理每种故障类型
    for fault_type in FAULT_TYPES.keys():
        print(f"\n处理故障类型: {fault_type}")
        
        # 提取该故障类型的权重
        weights = {
            'CTGAN': summary_df.loc[fault_type, 'Weight_CTGAN'],
            'DCGAN': summary_df.loc[fault_type, 'Weight_DCGAN'],
            'infoGAN': summary_df.loc[fault_type, 'Weight_infoGAN'],
            'LSGAN': summary_df.loc[fault_type, 'Weight_LSGAN'],
            'WGAN_GP': summary_df.loc[fault_type, 'Weight_WGAN_GP']
        }
        
        # 计算总权重并归一化
        total_weight = sum(w for w in weights.values() if w > 0)
        normalized_weights = {k: w/total_weight for k, w in weights.items() if w > 0}
        
        # 设置总样本数
        total_samples = 1000
        
        # 根据权重分配样本数量
        samples_per_gan = {k: int(w * total_samples) for k, w in normalized_weights.items()}
        
        # 处理舍入误差
        remaining_samples = total_samples - sum(samples_per_gan.values())
        if remaining_samples > 0:
            # 将剩余样本分配给权重最大的GAN
            max_weight_gan = max(normalized_weights.items(), key=lambda x: x[1])[0]
            samples_per_gan[max_weight_gan] += remaining_samples
        
        print("\n样本分配:")
        for gan_type, n_samples in samples_per_gan.items():
            print(f"{gan_type}: {n_samples} 样本 (权重: {normalized_weights[gan_type]:.4f})")
        
        # 存储所有生成的样本
        all_samples = []
        
        # 生成每个GAN的样本
        for gan_type, n_samples in samples_per_gan.items():
            if n_samples > 0:
                try:
                    # 构建模型文件名
                    model_name = f"{'wgan_gp' if gan_type == 'WGAN_GP' else gan_type.lower()}_generator_{fault_type}.pth"
                    model_path = os.path.join(model_base_path, model_name)
                    
                    # 加载生成器并生成样本（现在传入device参数）
                    generator, save_dict = load_gan_generator(gan_type, model_path, device)
                    samples = generate_samples(generator, gan_type, n_samples, device)
                    
                    # 反归一化
                    samples = denormalize_data(
                        samples,
                        save_dict['min_vals'],
                        save_dict['max_vals'],
                        gan_type  # 传入生成器类型
                    )
                    
                    # 添加到样本列表
                    all_samples.append(samples)
                    print(f"成功生成 {gan_type} 的 {n_samples} 个样本")
                    
                except Exception as e:
                    print(f"处理 {gan_type} 时出错: {str(e)}")
                    continue
        
        # 合并并保存样本
            if all_samples:
                combined_data = np.vstack(all_samples)
                
                # 保存生成的数据
                output_file = os.path.join(output_dir, f'weighted_combined_{fault_type}.csv')
                df = pd.DataFrame(combined_data, columns=['RP_Before_MPP', 'RP_After_MPP'])
                df.to_csv(output_file, index=False)
                
                print(f"\n保存数据到: {output_file}")
                print(f"数据范围:")
                print(f"RP_Before_MPP: [{combined_data[:, 0].min():.4f}, {combined_data[:, 0].max():.4f}]")
                print(f"RP_After_MPP: [{combined_data[:, 1].min():.4f}, {combined_data[:, 1].max():.4f}]") 
            
            # 保存权重信息
            weight_info = pd.DataFrame({
                'GAN_Type': list(samples_per_gan.keys()),
                'Weight': [normalized_weights[gan] for gan in samples_per_gan.keys()],
                'Samples': list(samples_per_gan.values())
            })
            weight_info.to_csv(os.path.join(output_dir, f'weights_{fault_type}.csv'), index=False)
    
    print("\n所有数据生成完成!")

if __name__ == "__main__":
    main()
