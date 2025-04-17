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

def generate_weighted_data_by_fault(weights_file, fault_type, model_base_path, output_dir, total_samples=1000):
    """根据权重比例生成不同数量的样本并合并"""
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
    
    # 计算权重总和并归一化
    total_weight = sum(w for w in weights.values() if w > 0)
    normalized_weights = {k: w/total_weight for k, w in weights.items() if w > 0}
    
    # 计算每个GAN应该生成的样本数量
    samples_per_gan = {k: int(w * total_samples) for k, w in normalized_weights.items()}
    
    # 处理舍入误差，确保总样本数正确
    remaining_samples = total_samples - sum(samples_per_gan.values())
    if remaining_samples > 0:
        # 将剩余的样本分配给权重最大的GAN
        max_weight_gan = max(normalized_weights.items(), key=lambda x: x[1])[0]
        samples_per_gan[max_weight_gan] += remaining_samples
    
    print("\n各GAN生成样本数量:")
    for gan_type, n_samples in samples_per_gan.items():
        print(f"{gan_type}: {n_samples} 样本 (权重: {normalized_weights[gan_type]:.4f})")
    
    # 存储合并数据
    all_samples = []
    
    # 处理每个GAN类型
    for gan_type, n_samples in samples_per_gan.items():
        if n_samples > 0:
            print(f"\n处理 {gan_type}, 生成 {n_samples} 个样本")
            
            # 构建模型文件名
            model_name = f"{'wgan_gp' if gan_type == 'WGAN_GP' else gan_type.lower()}_generator_{fault_type}.pth"
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

                
                # 添加到样本列表
                all_samples.append(samples)
                print(f"成功处理 {gan_type}")
                
            except Exception as e:
                print(f"处理模型 {model_path} 时出错: {str(e)}")
                continue
    
    # 合并所有样本
    if all_samples:
        combined_data = np.vstack(all_samples)
        
        # 保存合并后的数据
        output_file = os.path.join(output_dir, f'weighted_combined_{fault_type}.csv')
        df = pd.DataFrame(combined_data, columns=['RP_Before_MPP', 'RP_After_MPP'])
        df.to_csv(output_file, index=False)
        print(f"\n已保存数据到: {output_file}")
        print(f"数据范围:")
        print(f"RP_Before_MPP: [{combined_data[:, 0].min():.4f}, {combined_data[:, 0].max():.4f}]")
        print(f"RP_After_MPP: [{combined_data[:, 1].min():.4f}, {combined_data[:, 1].max():.4f}]")
        
        # 保存采样信息
        sampling_info = pd.DataFrame({
            'GAN_Type': list(samples_per_gan.keys()),
            'Original_Weight': [weights[gan] for gan in samples_per_gan.keys()],
            'Normalized_Weight': [normalized_weights[gan] for gan in samples_per_gan.keys()],
            'Samples_Generated': list(samples_per_gan.values())
        })
        sampling_file = os.path.join(output_dir, f'sampling_info_{fault_type}.csv')
        sampling_info.to_csv(sampling_file, index=False)
        
        print("\n采样信息:")
        print(sampling_info)
        
def generate_optimization_summary(all_results):
    """生成优化结果的总结报告，选择归一化评价指标总和最大的解"""
    summary_df = pd.DataFrame()
    
    for fault_type, results in all_results.items():
        # 归一化四个评价指标
        normalized_scores = results.copy()
        
        # MMD和Wasserstein是越小越好，需要反向归一化
        for col in ['MMD', 'Wasserstein']:
            min_val = results[col].min()
            max_val = results[col].max()
            if max_val - min_val != 0:
                normalized_scores[col] = (max_val - results[col]) / (max_val - min_val)
            else:
                normalized_scores[col] = 1
        
        # Coverage和ModeScore是越大越好，直接归一化
        for col in ['Coverage', 'ModeScore']:
            min_val = results[col].min()
            max_val = results[col].max()
            if max_val - min_val != 0:
                normalized_scores[col] = (results[col] - min_val) / (max_val - min_val)
            else:
                normalized_scores[col] = 1
        
        # 计算总分
        total_scores = normalized_scores[['MMD', 'Coverage', 'Wasserstein', 'ModeScore']].sum(axis=1)
        
        # 找出总分最高的解的索引
        best_index = total_scores.idxmax()
        
        # 记录最佳权重和原始指标值
        gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
        for gan_type in gan_types:
            summary_df.loc[fault_type, f'Weight_{gan_type}'] = results.loc[best_index, f'Weight_{gan_type}']
        
        # 记录原始指标值
        summary_df.loc[fault_type, 'MMD'] = results.loc[best_index, 'MMD']
        summary_df.loc[fault_type, 'Coverage'] = results.loc[best_index, 'Coverage']
        summary_df.loc[fault_type, 'Wasserstein'] = results.loc[best_index, 'Wasserstein']
        summary_df.loc[fault_type, 'ModeScore'] = results.loc[best_index, 'ModeScore']
        summary_df.loc[fault_type, 'Total_Score'] = total_scores[best_index]
    
    # 保存总结报告
    summary_path = os.path.join(
        r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_results",
        'optimization_summary.csv'
    )
    summary_df.to_csv(summary_path)
    print(f"\n优化总结报告已保存到: {summary_path}")
    
    # 打印每种故障类型的最优解
    for fault_type in summary_df.index:
        print(f"\n{fault_type} 故障类型的最优解:")
        print("权重:")
        for gan_type in gan_types:
            weight = summary_df.loc[fault_type, f'Weight_{gan_type}']
            print(f"{gan_type}: {weight:.4f}")
        print("评价指标:")
        print(f"MMD: {summary_df.loc[fault_type, 'MMD']:.4f}")
        print(f"Coverage: {summary_df.loc[fault_type, 'Coverage']:.4f}")
        print(f"Wasserstein: {summary_df.loc[fault_type, 'Wasserstein']:.4f}")
        print(f"ModeScore: {summary_df.loc[fault_type, 'ModeScore']:.4f}")
        print(f"总分: {summary_df.loc[fault_type, 'Total_Score']:.4f}")

def MMD(G, X_real=None):
    """计算MMD距离"""
    # 参考EPSOGAN中的实现
    from EPSOGAN import MMD as mmd_func
    return mmd_func(G, X_real)

def Coverage(G, X_real=None):
    """计算Coverage分数"""
    from EPSOGAN import Coverage as coverage_func
    return coverage_func(G, X_real)

def Wasserstein(G, X_real=None):
    """计算Wasserstein距离"""
    from EPSOGAN import Wasserstein as wasserstein_func
    return wasserstein_func(G, X_real)

def ModeScore(G, X_real=None):
    """计算Mode Score"""
    from EPSOGAN import ModeScore as mode_score_func
    return mode_score_func(G, X_real)

def main():
    # 设置路径
    model_base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\gan_model"
    mopso_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_results"
    output_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\weighted_gan_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 故障类型列表
    fault_types = [
        'aging', 'normal', 'open', 'shading', 
        'short', 'aging_open', 'aging_short'
    ]
    
    print("开始生成加权数据...")
    
    # 存储所有评估结果
    all_results = {}
    
    # 处理每种故障类型
    for fault_type in fault_types:
        weights_file = os.path.join(mopso_dir, f'mopso_results_{fault_type}.csv')
        if os.path.exists(weights_file):
            # 读取权重文件的所有解
            results_df = pd.read_csv(weights_file)
            
            # 为每个解生成并评估数据
            eval_results = pd.DataFrame()
            for idx in range(len(results_df)):
                # 创建临时权重文件
                temp_weights = results_df.iloc[[idx]]
                temp_weights_file = os.path.join(mopso_dir, f'temp_weights_{fault_type}.csv')
                temp_weights.to_csv(temp_weights_file, index=False)
                
                try:
                    # 生成加权数据
                    generate_weighted_data_by_fault(
                        weights_file=temp_weights_file,
                        fault_type=fault_type,
                        model_base_path=model_base_path,
                        output_dir=output_dir,
                        total_samples=1000
                    )
                    
                    # 加载生成的数据和真实数据
                    gen_data = pd.read_csv(os.path.join(output_dir, f'weighted_combined_{fault_type}.csv')).values
                    real_data_file = FAULT_TYPES[fault_type]  # 使用映射获取正确的文件名
                    real_data = pd.read_csv(os.path.join(
                        r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\LD",
                        real_data_file
                    )).values
                    
                    # 计算评价指标
                    mmd_score = MMD(gen_data, real_data)
                    coverage_score = Coverage(gen_data, real_data)
                    wasserstein_score = Wasserstein(gen_data, real_data)
                    mode_score = ModeScore(gen_data, real_data)
                    
                    # 保存结果
                    for col in results_df.columns:
                        eval_results.loc[idx, col] = results_df.loc[idx, col]
                    eval_results.loc[idx, 'MMD'] = mmd_score
                    eval_results.loc[idx, 'Coverage'] = coverage_score
                    eval_results.loc[idx, 'Wasserstein'] = wasserstein_score
                    eval_results.loc[idx, 'ModeScore'] = mode_score
                    
                except Exception as e:
                    print(f"评估解 {idx} 时出错: {str(e)}")
                    continue
                
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_weights_file):
                        os.remove(temp_weights_file)
            
            if not eval_results.empty:
                all_results[fault_type] = eval_results
        else:
            print(f"\n警告: 找不到故障类型 {fault_type} 的权重文件: {weights_file}")
    
    # 生成优化总结报告
    if all_results:
        generate_optimization_summary(all_results)
    
    print("\n所有数据生成和评估完成!")



if __name__ == "__main__":
    main()
