import pandas as pd
import numpy as np
import os
import umap
from sklearn.preprocessing import StandardScaler

def extract_iv_features(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, skiprows=1)
    
    num_columns = len(df.columns)
    num_curves = num_columns // 2
    print(f"共有 {num_curves} 组IV特性曲线数据...")
    
    # 存储所有曲线的数据点
    all_curves = []
    
    for i in range(num_curves):
        try:
            # 获取第i组的电压和电流数据
            voltage = np.array(df.iloc[:, i*2], dtype=float)
            current = np.array(df.iloc[:, i*2+1], dtype=float)
            
            # 数据清洗
            mask = (~np.isnan(voltage) & ~np.isnan(current) & 
                   (np.abs(voltage) > 1e-10) & (np.abs(current) > 1e-10))
            voltage = voltage[mask]
            current = current[mask]
            
            if len(voltage) < 6:
                print(f"第 {i+1} 组数据点不足")
                continue
                
            # 确保电压是递减顺序
            sort_idx = np.argsort(voltage)[::-1]
            voltage = voltage[sort_idx]
            current = current[sort_idx]
            
            # 组合电压和电流数据
            curve_data = np.column_stack((voltage, current))
            all_curves.append(curve_data)
            
        except Exception as e:
            print(f"第 {i+1} 组处理失败: {str(e)}")
            continue
    
    # 确保有足够的数据进行UMAP降维
    if not all_curves:
        raise ValueError("没有有效的IV曲线数据")
    
    # 计算每条曲线的额外特征
    enhanced_curves = []
    fixed_length = 200  # 减少采样点数以提高性能
    
    for curve in all_curves:
        voltage, current = curve[:, 0], curve[:, 1]
        
        # 重采样到固定长度
        indices = np.linspace(0, len(voltage)-1, fixed_length)
        voltage_resampled = np.interp(indices, np.arange(len(voltage)), voltage)
        current_resampled = np.interp(indices, np.arange(len(current)), current)
        
        # 计算功率
        power = voltage_resampled * current_resampled
        mpp_idx = np.argmax(power)
        
        # 安全地计算特征
        try:
            max_power = np.max(power)
            v_mpp = voltage_resampled[mpp_idx]
            i_mpp = current_resampled[mpp_idx]
            v_oc = voltage_resampled[0]
            i_sc = current_resampled[-1]
            
            # 计算填充因子，添加安全检查
            ff = max_power / (v_oc * i_sc) if abs(v_oc * i_sc) > 1e-10 else 0
            
            # 安全地计算斜率
            eps = 1e-10
            if mpp_idx > 0:  # 确保有足够的点计算斜率
                voltage_diff_before = np.diff(voltage_resampled[:mpp_idx+1]) + eps
                current_diff_before = np.diff(current_resampled[:mpp_idx+1])
                slope_before = np.mean(current_diff_before / voltage_diff_before)
            else:
                slope_before = 0
                
            if mpp_idx < len(voltage_resampled) - 1:
                voltage_diff_after = np.diff(voltage_resampled[mpp_idx:]) + eps
                current_diff_after = np.diff(current_resampled[mpp_idx:])
                slope_after = np.mean(current_diff_after / voltage_diff_after)
            else:
                slope_after = 0
                
            # 组合特征
            curve_features = np.array([
                v_oc, i_sc, v_mpp, i_mpp, max_power, ff,
                slope_before, slope_after
            ], dtype=np.float32)  # 使用float32减少内存使用
            
            # 使用float32类型减少内存使用
            enhanced_curve = np.concatenate([
                voltage_resampled.astype(np.float32),
                current_resampled.astype(np.float32),
                curve_features
            ])
            enhanced_curves.append(enhanced_curve)
            
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            continue
    
    if not enhanced_curves:
        raise ValueError("没有有效的特征数据")
    
    # 转换为numpy数组
    X = np.array(enhanced_curves, dtype=np.float32)
    
    return X

def process_all_csv_files(base_path):
    """
    处理指定目录下所有包含'IV'的CSV文件，按标签进行聚类
    """
    # 定义需要处理的子目录
    subdirs = ['ISD', 'BSD', 'LD']
    
    # 存储所有数据和标签
    all_features = []
    all_labels = []
    file_mapping = {}  # 保存文件名到处理后数据的映射
    
    for subdir in subdirs:
        input_dir = os.path.join(base_path, "data_normalize", subdir)
        
        if not os.path.exists(input_dir):
            print(f"警告: 目录 {input_dir} 不存在，跳过处理")
            continue
        
        # 获取目录下所有包含'IV'的CSV文件
        csv_files = [f for f in os.listdir(input_dir) 
                    if f.endswith('.csv') and 'IV' in f]
        
        if not csv_files:
            print(f"警告: 在 {input_dir} 中没有找到包含'IV'的CSV文件")
            continue
            
        print(f"\n开始处理 {subdir} 目录下的IV曲线文件...")
        
        # 处理每个CSV文件
        for csv_file in csv_files:
            try:
                print(f"\n正在处理文件: {csv_file}")
                input_file = os.path.join(input_dir, csv_file)
                
                # 提取IV曲线特征
                curves_features = extract_iv_features(input_file)  # 提取但不进行UMAP降维
                
                # 确定标签
                if 'aging_short' in csv_file.lower():
                    label = 0  # DOA+SC
                elif 'aging' in csv_file.lower():
                    label = 1  # DOA
                elif 'normal' in csv_file.lower():
                    label = 2  # Normal
                elif 'short' in csv_file.lower():
                    label = 3  # SC
                elif 'open' in csv_file.lower():
                    label = 4  # Open
                elif 'shading' in csv_file.lower():
                    label = 5  # Shading
                
                # 保存特征和标签
                all_features.extend(curves_features)
                all_labels.extend([label] * len(curves_features))
                file_mapping[csv_file] = (len(all_features) - len(curves_features), len(all_features))
                
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {str(e)}")
    
    # 将所有特征转换为numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # 标准化
    X_scaled = StandardScaler().fit_transform(X)
    
    # 配置UMAP参数
    reducer = umap.UMAP(
        n_neighbors=50,      
        min_dist=0.05,       
        n_components=2,      
        metric='euclidean',  
        random_state=42,    # 添加随机种子以保持结果一致
        n_epochs=1000,      # 增加训练轮数
        learning_rate=0.5,   
        init='spectral',     
        target_weight=0.8,   
        target_metric='categorical',
        densmap=True,        
        output_dens=False,   # 关闭密度输出
        dens_lambda=2.0,     
        n_jobs=-1           
    )
    
    # 使用UMAP进行降维
    print("开始UMAP降维...")
    embedding = reducer.fit_transform(X_scaled, y)
    print("UMAP降维完成")
    print(f"降维结果形状: {embedding.shape}")
    
    
    # 创建特征结果存储目录
    features_base_dir = os.path.join(base_path, "features")
    if not os.path.exists(features_base_dir):
        os.makedirs(features_base_dir)
    
    # 保存结果
    for subdir in subdirs:
        output_dir = os.path.join(features_base_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        for csv_file, (start_idx, end_idx) in file_mapping.items():
            if csv_file.endswith(f'_{subdir}_corrected.csv'):
                try:
                    # 提取当前文件的降维结果
                    current_embedding = embedding[start_idx:end_idx]
                    
                    # 创建结果文件名
                    output_filename = f"rp_features_{os.path.splitext(csv_file)[0]}.csv"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 保存特征到CSV文件
                    result_df = pd.DataFrame(current_embedding, columns=['UMAP_Feature1', 'UMAP_Feature2'])
                    result_df.to_csv(output_path, index=False)
                    
                    print(f"特征提取完成，结果已保存到 {output_path}")
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {str(e)}")

if __name__ == "__main__":
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code"
    process_all_csv_files(base_path)