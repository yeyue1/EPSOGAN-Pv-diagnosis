import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import itertools

def load_test_data(file_paths):
    all_data = []
    all_labels = []
    
    for label, file_path in enumerate(file_paths):
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 获取特征数据
            features = df.values
            
            # 添加数据和标签
            all_data.extend(features)
            all_labels.extend([label] * len(features))
            
            print(f"成功读取文件 {file_path}，获取了 {len(features)} 个样本")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return np.array(all_data), np.array(all_labels)

def predict(model_dict, data):
    # 使用保存的scaler进行数据标准化
    scaled_data = model_dict['scaler'].transform(data)
    # 使用模型进行预测
    return model_dict['model'].predict(scaled_data)

def plot_confusion_matrix(y_true, y_pred, save_path, fault_types):
    """绘制混淆矩阵，分别显示数值和百分比"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(15, 12))
    
    # 创建热图，使用较浅的蓝色色系
    sns.heatmap(
        cm,
        cmap='Blues',
        xticklabels=fault_types,
        yticklabels=fault_types,
        square=True,
        cbar=True,
        vmax=cm.max() * 1.2,
        vmin=0
    )
    
    # 添加数值和百分比标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color_value = cm[i, j] / cm.max()
            text_color = 'white' if color_value > 0.5 else 'black'
            
            # 在单元格中心偏上位置添加数值
            plt.text(
                j + 0.5,
                i + 0.35,
                str(cm[i, j]),
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                color=text_color,
            )
            
            # 在单元格中心偏下位置添加百分比
            plt.text(
                j + 0.5,
                i + 0.65,
                f'({cm_percentages[i, j]:.1f}%)',
                ha='center',
                va='center',
                fontsize=10,
                color=text_color,
            )
    
    # 调整标签位置和大小
    plt.xticks(
        np.arange(len(fault_types)) + 0.5,
        fault_types,
        rotation=0,
        ha='center',
        va='top',
        fontsize=12
    )
    plt.yticks(
        np.arange(len(fault_types)) + 0.5,
        fault_types,
        rotation=0,
        ha='right',
        va='center',
        fontsize=12
    )
    
    # 增加对标签的额外调整
    plt.gca().set_xticklabels(fault_types, style='normal')
    plt.gca().set_yticklabels(fault_types, style='normal')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close()

def main():
    r''' 
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\ISD"
    # 文件路径
    file_names = [
        'rp_features_pv_aging_IV_ISD_corrected.csv',
        'rp_features_pv_normal_IV_ISD_corrected.csv',
        'rp_features_pv_open_circuit_IV_ISD_corrected.csv',
        'rp_features_pv_shading_IV_ISD_corrected.csv',
        'rp_features_pv_short_circuit_IV_ISD_corrected.csv',
        'rp_features_aging_open_data_IV_ISD_corrected.csv',
        'rp_features_aging_short_data_IV_ISD_corrected.csv'
    ]
    r'''
    
    # 修改文件路径为RP特征文件
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\generated_samples"
    # 文件路径
    file_names = [
        'wgan_aging.csv',
        'wgan_normal.csv',
        'wgan_open.csv',
        'wgan_shading.csv',
        'wgan_short.csv',
        'wgan_aging_open.csv',
        'wgan_aging_short.csv',
    ]
    r''' 
    # 修改文件路径为RP特征文件
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\weighted_gan_results"
    # 文件路径
    file_names = [
        'weighted_combined_aging.csv',
        'weighted_combined_normal.csv',
        'weighted_combined_open.csv',
        'weighted_combined_shading.csv', 
        'weighted_combined_short.csv',
        'weighted_combined_aging_open.csv',
        'weighted_combined_aging_short.csv',   
    ]
    r'''
    # 构建完整文件路径
    file_paths = [os.path.join(base_path, fname) for fname in file_names]
    
    # 故障类型标签
    fault_types = [
        'Aging', 'Normal', 'Open Circuit', 'Shading',
        'Short Circuit', 'Aging-Open', 'Aging-Short'
    ]
    
    # 加载模型
    model_path = 'C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/predict/best_model/svm_ISD_RP_best_model.pkl'
    model_dict = joblib.load(model_path)
    
    try:
        # 加载测试数据
        print("开始加载测试数据...")
        test_data, test_labels = load_test_data(file_paths)
        print(f"成功加载测试数据: {len(test_data)} 个样本")
        
        # 预测
        predictions = predict(model_dict, test_data)
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, target_names=fault_types))
        
        # 绘制混淆矩阵并保存
        plot_confusion_matrix(test_labels, predictions, 'SVM_confusion_matrix_ISD_rp.jpg', fault_types)
        print("\n混淆矩阵已保存为 SVM_confusion_matrix_ISD_rp.jpg")
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()