import pandas as pd
import numpy as np
from EPSOGAN import MMD, Coverage, Wasserstein, ModeScore, f_GCaculate
import os
from pathlib import Path

# 在GAN_TYPES中添加加权GAN
GAN_TYPES = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP', 'Weighted_GAN']

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

def load_real_data(filepath):
    """加载真实数据"""
    df = pd.read_csv(filepath)
    return df.values

def load_gan_data(base_path, gan_type, fault_type):
    """加载指定GAN和故障类型的生成数据"""
    if gan_type == 'Weighted_GAN':
        # 加载加权GAN的数据
        weighted_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\weighted_gan_results"
        filepath = os.path.join(weighted_path, f'weighted_combined_{fault_type}.csv')
    else:
        # 加载其他GAN的数据
        gan_prefix_map = {
            'DCGAN': 'dcgan',
            'CTGAN': 'ctgan',
            'infoGAN': 'infogan',
            'LSGAN': 'lsgan',
            'WGAN_GP': 'wgan'
        }
        filename = f"{gan_prefix_map[gan_type]}_{fault_type}.csv"
        filepath = os.path.join(base_path, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    print(f"正在加载 {gan_type} {fault_type} 数据从: {filepath}")
    return pd.read_csv(filepath).values

def evaluate_gan(gan_data, real_data):
    """评估单个GAN的性能"""
    try:
        # 确保数据格式正确
        gan_data = np.array(gan_data, dtype=float)
        real_data = np.array(real_data, dtype=float)
        
        print(f"计算评估指标... 数据形状: GAN={gan_data.shape}, Real={real_data.shape}")
        
        results = {
            'MMD': float(MMD(gan_data, real_data)),
            'Coverage': float(Coverage(gan_data, real_data)),
            'Wasserstein': float(Wasserstein(gan_data, real_data)),
            'ModeScore': float(ModeScore(gan_data, real_data))
        }
        
        # 验证结果
        for key, value in results.items():
            if np.isnan(value) or np.isinf(value):
                print(f"警告: {key} 的值无效: {value}")
                results[key] = 0.0
            else:
                print(f"成功计算 {key}: {value:.4f}")
                
        return results
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        return None

def evaluate_all_gans():
    """评估所有GAN在所有故障类型上的性能"""
    try:
        base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\generated_samples"
        real_data_base = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\LD"
        output_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\evaluation_results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储所有评估结果
        all_results = {}
        
        # 对每种故障类型进行评估
        for fault_type, real_file in FAULT_TYPES.items():
            print(f"\n评估故障类型: {fault_type}")
            
            # 加载真实数据
            real_data_path = os.path.join(real_data_base, real_file)
            real_data = load_real_data(real_data_path)
            
            fault_results = {}
            # 评估每种GAN
            for gan_type in GAN_TYPES:
                try:
                    print(f"\n处理 {gan_type} 在 {fault_type} 故障上...")
                    gan_data = load_gan_data(base_path, gan_type, fault_type)
                    results = evaluate_gan(gan_data, real_data)
                    
                    if results:
                        fault_results[gan_type] = results
                    
                except Exception as e:
                    print(f"处理 {gan_type} {fault_type} 时出错: {str(e)}")
                    continue
            
            # 保存该故障类型的评估结果
            if fault_results:
                results_df = pd.DataFrame(fault_results).T
                
                # 计算F_G值
                for gan_type in results_df.index:
                    scores = results_df.loc[gan_type]
                    f_g = f_GCaculate(
                        scores['MMD'],
                        scores['Coverage'],
                        scores['Wasserstein'],
                        scores['ModeScore']
                    )
                    results_df.loc[gan_type, 'F_G'] = f_g
                
                # 保存结果
                output_file = os.path.join(output_dir, f'evaluation_results_{fault_type}.csv')
                results_df.to_csv(output_file)
                print(f"\n{fault_type} 故障的评估结果已保存到: {output_file}")
                
                all_results[fault_type] = results_df
        
        # 生成总体评估报告
        generate_summary_report(all_results, output_dir)
        
    except Exception as e:
        print(f"评估过程出错: {str(e)}")
        raise

def generate_summary_report(all_results, output_dir):
    """生成总体评估报告"""
    summary_df = pd.DataFrame()
    comparison_df = pd.DataFrame()
    
    for fault_type, results in all_results.items():
        # 为每个评估指标计算最佳GAN
        for metric in ['MMD', 'Coverage', 'Wasserstein', 'ModeScore', 'F_G']:
            if metric in ['Coverage', 'ModeScore', 'F_G']:
                best_gan = results[metric].idxmax()
                best_value = results[metric].max()
                weighted_value = results.loc['Weighted_GAN', metric]
            else:
                best_gan = results[metric].idxmin()
                best_value = results[metric].min()
                weighted_value = results.loc['Weighted_GAN', metric]
                
            summary_df.loc[fault_type, f'Best_{metric}_GAN'] = best_gan
            summary_df.loc[fault_type, f'Best_{metric}_Value'] = best_value
            summary_df.loc[fault_type, f'Weighted_{metric}_Value'] = weighted_value
            
            # 计算加权GAN相对于最佳GAN的性能比较
            if metric in ['Coverage', 'ModeScore', 'F_G']:
                relative_performance = weighted_value / best_value * 100
            else:
                relative_performance = best_value / weighted_value * 100
            comparison_df.loc[fault_type, f'{metric}_Performance'] = relative_performance
    
    # 保存总结报告
    summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file)
    print(f"\n总体评估报告已保存到: {summary_file}")
    
    # 保存性能比较报告
    comparison_file = os.path.join(output_dir, 'weighted_gan_comparison.csv')
    comparison_df.to_csv(comparison_file)
    print(f"加权GAN性能比较报告已保存到: {comparison_file}")
    
    # 打印性能总结
    print("\n加权GAN性能总结:")
    print("相对性能百分比 (100%表示与最佳性能相同):")
    print(comparison_df.mean().round(2))

if __name__ == "__main__":
    evaluate_all_gans()
