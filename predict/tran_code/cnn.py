import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
 # 模型保存路径
model_save_path = r"C:/Users/yeyue/Desktop/实验室工作用/论文2/Paper_Code/predict/best_model/cnn_BSD_RP_best_model.pth"


# 修改数据加载函数
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
            
            # 获取两列特征数据
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

# 自定义数据集类
class PVDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 修改模型结构
class PVCNN(nn.Module):
    def __init__(self):
        super(PVCNN, self).__init__()
        
        # 全连接层
        self.fc1 = nn.Linear(2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 7)  # 7类故障
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=50, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        print('--------------------')
        
        if scheduler is not None:
            scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        # 提前停止条件
        if epoch > 50 and best_val_acc > 99.5:
            print("Reached high accuracy, stopping early!")
            break

# 修改文件路径
def main():
    r''' 
    # 基础路径
    base_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\LD"
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 基础路径不存在: {base_path}")
        return
        
    # 文件路径
    file_names = [
        'rp_features_pv_aging_IV_LD_corrected.csv',
        'rp_features_pv_normal_IV_LD_corrected.csv',
        'rp_features_pv_open_circuit_IV_LD_corrected.csv',
        'rp_features_pv_shading_IV_LD_corrected.csv',
        'rp_features_pv_short_circuit_IV_LD_corrected.csv',
        'rp_features_aging_open_data_IV_LD_corrected.csv',
        'rp_features_aging_short_data_IV_LD_corrected.csv'
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
        
        # 创建数据加载器
        train_dataset = PVDataset(X_train, y_train)
        test_dataset = PVDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 初始化模型和训练参数
        model = PVCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=5, 
                                                       verbose=True)
        
        # 检查是否可以使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        
       
        
        # 训练模型
        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler=scheduler, num_epochs=100, device=device)
        
        # 加载最佳模型并进行测试
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Final Test Accuracy: {test_acc:.2f}%')
    
    except Exception as e:
        print(f"运行主程序时出错: {str(e)}")

if __name__ == '__main__':
    main()