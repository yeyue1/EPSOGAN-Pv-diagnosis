import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 模型保存路径
model_save_path = r"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/predict/best_model/bayes_ISD_RP_best_model.pkl"

def load_data(file_paths):
    all_data = []
    all_labels = []
    
    for label, file_path in enumerate(file_paths):
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")
                continue
                
            # 读取CSV文件，跳过第一行表头
            print(f"正在读取文件: {file_path}")
            df = pd.read_csv(file_path, skiprows=1)
            
            # 获取特征数据
            features = df.values
            
            # 添加数据和标签
            all_data.extend(features)
            all_labels.extend([label] * len(features))
            
            print(f"成功读取文件，获取了 {len(features)} 个样本")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    if not all_data:
        raise ValueError("没有成功读取任何数据！请检查文件路径。")
        
    return np.array(all_data), np.array(all_labels)

def train_model(X_train, X_test, y_train, y_test):
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 初始化贝叶斯模型
    bayes_model = GaussianNB(
        priors=None,            # 类别的先验概率，默认为None表示按类别频率计算
        var_smoothing=1e-9      # 方差平滑参数，用于处理零方差
    )
    
    # 训练模型
    print("开始训练贝叶斯模型...")
    bayes_model.fit(X_train_scaled, y_train)
    
    # 在训练集上评估
    train_pred = bayes_model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"训练集准确率: {train_acc*100:.2f}%")
    
    # 在测试集上评估
    test_pred = bayes_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"测试集准确率: {test_acc*100:.2f}%")
    
    # 保存最佳模型
    model_dict = {
        'model': bayes_model,
        'scaler': scaler
    }
    joblib.dump(model_dict, model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 打印详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, test_pred))
    
    return bayes_model, scaler

def main():
    r''' 
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\BSD"
    # 文件路径
    file_names = [
        'rp_features_pv_aging_IV_BSD_corrected.csv',
        'rp_features_pv_normal_IV_BSD_corrected.csv',
        'rp_features_pv_open_circuit_IV_BSD_corrected.csv',
        'rp_features_pv_shading_IV_BSD_corrected.csv',
        'rp_features_pv_short_circuit_IV_BSD_corrected.csv',
        'rp_features_aging_open_data_IV_BSD_corrected.csv',
        'rp_features_aging_short_data_IV_BSD_corrected.csv'
    ]
    r'''
    r'''
    # 修改文件路径为RP特征文件
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\generated_samples"
    # 文件路径
    file_names = [
        'dcgan_aging.csv',
        'dcgan_normal.csv',
        'dcgan_open.csv',
        'dcgan_shading.csv',
        'dcgan_short.csv',
        'dcgan_aging_open.csv',
        'dcgan_aging_short.csv',
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
    
    
    # 构建完整文件路径
    file_paths = [os.path.join(base_path, fname) for fname in file_names]
    
    try:
        # 加载数据
        print("开始加载数据...")
        data, labels = load_data(file_paths)
        print(f"成功加载数据: {len(data)} 个样本, {len(np.unique(labels))} 个类别")
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )
        
        # 训练模型
        bayes_model, scaler = train_model(X_train, X_test, y_train, y_test)
        
        # 加载保存的模型进行验证
        print("\n加载保存的模型进行验证...")
        loaded_model_dict = joblib.load(model_save_path)
        loaded_model = loaded_model_dict['model']
        loaded_scaler = loaded_model_dict['scaler']
        
        # 使用加载的模型进行预测
        X_test_scaled = loaded_scaler.transform(X_test)
        test_pred = loaded_model.predict(X_test_scaled)
        final_acc = accuracy_score(y_test, test_pred)
        print(f"加载的模型测试集准确率: {final_acc*100:.2f}%")
        
    except Exception as e:
        print(f"运行主程序时出错: {str(e)}")

if __name__ == '__main__':
    main()