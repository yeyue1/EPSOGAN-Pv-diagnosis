import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from torch.serialization import add_safe_globals
from pandas.core.series import Series

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入生成器
from WGAN_GP import Generator as WGAN_Generator
from CTGAN import Generator as CTGAN_Generator
from DCGAN import Generator as DCGAN_Generator
from LSGAN import Generator as LSGAN_Generator
from infoGAN import Generator as InfoGAN_Generator
# 导入扩散模型相关类
from diffusion import DiffusionModel, NoisePredictor

# 定义故障类型映射
fault_types = {
    0: 'aging',        # 老化故障
    1: 'normal',       # 正常工作
    2: 'open',         # 开路故障
    3: 'shading',      # 遮挡故障
    4: 'short',        # 短路故障
    5: 'aging_open',   # 老化-开路复合故障
    6: 'aging_short'   # 老化-短路复合故障
}

# 添加Series到安全全局变量列表
add_safe_globals([Series])

def load_gan_generators(generator_name, label, latent_dim=100, num_classes=7):
    """加载特定标签的生成器"""
    feature_dim = 2
    columns = ['RP_Before_MPP', 'RP_After_MPP']
    
    try:
        # 获取故障类型名称
        fault_name = fault_types[label]
        
        if generator_name == 'wgan':
            # 修改WGAN生成器初始化，添加num_classes参数
            generator = WGAN_Generator(latent_dim, feature_dim, num_classes).cuda()
            model_path = f"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/gan_model/wgan_gp_generator_{fault_name}.pth"
            
        elif generator_name == 'ctgan':
            generator = CTGAN_Generator(latent_dim, feature_dim, num_classes).cuda()
            model_path = f"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/gan_model/ctgan_generator_{fault_name}.pth"
            
        elif generator_name == 'dcgan':
            generator = DCGAN_Generator(latent_dim, feature_dim, num_classes).cuda()
            model_path = f"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/gan_model/dcgan_generator_{fault_name}.pth"
            
        elif generator_name == 'lsgan':
            generator = LSGAN_Generator(latent_dim, feature_dim, num_classes).cuda()
            model_path = f"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/gan_model/lsgan_generator_{fault_name}.pth"
            
        elif generator_name == 'infogan':
            generator = InfoGAN_Generator(62, num_classes, 2, feature_dim).cuda()
            model_path = f"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/gan_model/infogan_generator_{fault_name}.pth"
            
        elif generator_name == 'diffusion':
            # 加载扩散模型
            model_path = f"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/diffusion_model/diffusion_model_{fault_name}.pth"
            print(f"正在加载模型: {model_path}")
            save_dict = torch.load(model_path, weights_only=False)
            
            # 从保存的参数中重建噪声预测器
            noise_predictor = NoisePredictor(
                input_dim=save_dict['feature_dim'],
                categorical_dim=num_classes
            ).cuda()
            noise_predictor.load_state_dict(save_dict['model_state_dict'])
            
            # 创建完整的扩散模型
            generator = DiffusionModel(
                feature_dim=save_dict['feature_dim'],
                categorical_dim=num_classes,
                timesteps=save_dict.get('timesteps', 500)
            ).to('cuda')
            
            # 将训练好的噪声预测器赋给扩散模型
            generator.noise_predictor = noise_predictor
            generator.noise_predictor.eval()
            
            return generator, save_dict
            
        else:
            raise ValueError(f"不支持的生成器类型: {generator_name}")
        
        print(f"正在加载模型: {model_path}")
        save_dict = torch.load(model_path, weights_only=False)
        
        # 加载模型参数
        generator.load_state_dict(save_dict['generator_state_dict'])
        generator.eval()
        
        # 转换统计信息为Series或保持为numpy数组
        for key in ['min_vals', 'max_vals', 'mean_vals', 'std_vals']:
            if key in save_dict:
                save_dict[key] = save_dict[key][:2]  # 只保留前两个值
                
        return generator, save_dict
        
    except Exception as e:
        print(f"加载 {generator_name} 标签 {label} ({fault_name}) 的生成器时出错: {str(e)}")
        return None, None

def denormalize_data(data, min_vals, max_vals, generator_name):
    """根据不同GAN模型使用相应的反归一化方法"""
    data = np.array(data)
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    
    if generator_name.lower() in ['ctgan']:
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

def generate_specific_samples(generator_name, fault_name, label, num_samples=100, latent_dim=100, num_classes=7):
    """生成特定故障类型的样本"""
    generator, save_dict = load_gan_generators(generator_name, label, latent_dim, num_classes)
    
    if generator is None or save_dict is None:
        raise ValueError(f"无法加载 {generator_name} 故障类型 {fault_name} 的生成器")
    
    with torch.no_grad():
        if generator_name == 'wgan':
            # 修改WGAN的生成过程，添加标签
            z = torch.randn(num_samples, latent_dim).cuda()
            labels = torch.LongTensor([label] * num_samples).cuda()
            samples = generator(z, labels)  # 添加标签输入
        elif generator_name == 'infogan':
            z = torch.randn(num_samples, 62).cuda()
            categorical = torch.zeros(num_samples, num_classes).cuda()
            categorical[:, label] = 1
            continuous = torch.randn(num_samples, 2).cuda()
            samples = generator(z, categorical, continuous)
        elif generator_name == 'diffusion':
            # 扩散模型生成样本
            labels = torch.LongTensor([label] * num_samples).cuda()
            samples = generator.sample(num_samples, labels, torch.device('cuda'))
        else:
            z = torch.randn(num_samples, latent_dim).cuda()
            labels = torch.LongTensor([label] * num_samples).cuda()
            samples = generator(z, labels)
            
        # 获取生成的样本并转到CPU
        samples = samples.cpu().numpy()
        
        # 使用对应的反归一化方法
        samples = denormalize_data(
            samples,
            save_dict['min_vals'],
            save_dict['max_vals'],
            generator_name  # 传入生成器类型
        )
    # 修改保存路径，使用故障类型名称
    output_dir = "C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/generated_samples"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{generator_name}_{fault_name}.csv")
    
    df = pd.DataFrame(samples, columns=['RP_Before_MPP', 'RP_After_MPP'])
    df.to_csv(output_path, index=False)
    print(f"已生成 {generator_name} 的故障类型 {fault_name} 的样本，保存到: {output_path}")
    print(f"样本范围：")
    print(f"RP_Before_MPP - 最小值: {samples[:, 0].min():.2f}, 最大值: {samples[:, 0].max():.2f}")
    print(f"RP_After_MPP  - 最小值: {samples[:, 1].min():.2f}, 最大值: {samples[:, 1].max():.2f}")
    
    return samples

if __name__ == "__main__":
    try:
        # 设置参数
        generator_name = 'diffusion'  # 可以改为: 'wgan', 'ctgan', 'dcgan', 'lsgan', 'infogan', 'diffusion'
        num_samples = 1000  # 每个标签生成的样本数量
        
        print(f"\n使用 {generator_name} 生成器生成样本")
        
        # 为每个故障类型生成样本
        all_samples = []
        all_fault_names = []
        for label, fault_name in fault_types.items():
            print(f"\n正在生成 {fault_name} 故障的样本...")
            samples = generate_specific_samples(
                generator_name=generator_name,
                fault_name=fault_name,
                label=label,
                num_samples=num_samples
            )
            all_samples.append(samples)
            all_fault_names.extend([fault_name] * num_samples)
            print(f"{fault_name} 故障的样本生成完成，形状: {samples.shape}")
        
        # 保存所有样本到一个文件
        all_samples = np.concatenate(all_samples, axis=0)
        
        output_dir = "C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/gan_code/generated_samples"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{generator_name}_all_samples.csv")
        
        # 使用正确的列名和故障类型名称创建完整数据集
        df = pd.DataFrame(all_samples, columns=['RP_Before_MPP', 'RP_After_MPP'])
        df['fault_type'] = all_fault_names
        df.to_csv(output_path, index=False)
        print(f"\n所有样本已保存到: {output_path}")
        print(f"总样本数: {len(df)}, 形状: {all_samples.shape}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")