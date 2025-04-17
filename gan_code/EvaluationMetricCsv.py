import os
import pandas as pd

# 定义文件夹路径
input_folder = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\evaluation_results"
output_folder = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\comparison_tables"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 故障类型列表及其显示名称
fault_types = [
    ('aging', 'DOA'),
    ('normal', 'NS'),
    ('open', 'OC'),
    ('shading', 'PS'),
    ('short', 'SC'),
    ('aging_open', 'DOA+OC'),
    ('aging_short', 'DOA+SC')
]

# 创建四个空的DataFrame用于存储比较结果
mmd_df = pd.DataFrame(columns=['Fault_Type'])
coverage_df = pd.DataFrame(columns=['Fault_Type'])
wasserstein_df = pd.DataFrame(columns=['Fault_Type'])
modescore_df = pd.DataFrame(columns=['Fault_Type'])

# 添加故障类型列
for _, display_name in fault_types:
    mmd_df = mmd_df.append({'Fault_Type': display_name}, ignore_index=True)
    coverage_df = coverage_df.append({'Fault_Type': display_name}, ignore_index=True)
    wasserstein_df = wasserstein_df.append({'Fault_Type': display_name}, ignore_index=True)
    modescore_df = modescore_df.append({'Fault_Type': display_name}, ignore_index=True)

# 处理每个故障类型的评估文件
for file_name, display_name in fault_types:
    eval_file = os.path.join(input_folder, f"evaluation_results_{file_name}.csv")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(eval_file, index_col=0)
        
        # 将GAN列名重命名，将Weighted_GAN改为EPSOGAN
        df = df.rename(index={'Weighted_GAN': 'EPSOGAN'})
        
        # 获取当前故障类型的行索引
        row_idx = mmd_df[mmd_df['Fault_Type'] == display_name].index[0]
        
        # 填充四个指标表格
        for gan_type in df.index:
            gan_col = 'EPSOGAN' if gan_type == 'Weighted_GAN' else gan_type
            mmd_df.loc[row_idx, gan_col] = df.loc[gan_type, 'MMD']
            coverage_df.loc[row_idx, gan_col] = df.loc[gan_type, 'Coverage']
            wasserstein_df.loc[row_idx, gan_col] = df.loc[gan_type, 'Wasserstein']
            modescore_df.loc[row_idx, gan_col] = df.loc[gan_type, 'ModeScore']
    
    except Exception as e:
        print(f"处理 {file_name} 时出错: {e}")

# 保存比较表格
mmd_df.to_csv(os.path.join(output_folder, 'mmd_comparison.csv'), index=False)
coverage_df.to_csv(os.path.join(output_folder, 'coverage_comparison.csv'), index=False)
wasserstein_df.to_csv(os.path.join(output_folder, 'wasserstein_comparison.csv'), index=False)
modescore_df.to_csv(os.path.join(output_folder, 'modescore_comparison.csv'), index=False)

print("已生成四个指标比较表格CSV文件:")
print(f"1. MMD比较表: {os.path.join(output_folder, 'mmd_comparison.csv')}")
print(f"2. Coverage比较表: {os.path.join(output_folder, 'coverage_comparison.csv')}")
print(f"3. Wasserstein比较表: {os.path.join(output_folder, 'wasserstein_comparison.csv')}")
print(f"4. ModeScore比较表: {os.path.join(output_folder, 'modescore_comparison.csv')}")