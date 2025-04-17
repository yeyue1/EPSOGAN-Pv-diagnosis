import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这行
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

class CSVDataset(Dataset):
    def __init__(self, csv_folder, target_label=None):
        self.data = []
        self.labels = []
        self.min_vals = None
        self.max_vals = None
        
        # 故障类型与文件名的映射
        file_mapping = {
            0: 'rp_features_pv_aging_IV_BSD_corrected.csv',         # aging
            1: 'rp_features_pv_normal_IV_BSD_corrected.csv',        # normal
            2: 'rp_features_pv_open_circuit_IV_BSD_corrected.csv',  # open
            3: 'rp_features_pv_shading_IV_BSD_corrected.csv',       # shading
            4: 'rp_features_pv_short_circuit_IV_BSD_corrected.csv', # short
            5: 'rp_features_aging_open_data_IV_BSD_corrected.csv',  # aging_open
            6: 'rp_features_aging_short_data_IV_BSD_corrected.csv'  # aging_short
        }
        
        if target_label is not None:
            # 只加载指定标签的数据
            file_path = os.path.join(csv_folder, file_mapping[target_label])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到文件: {file_path}")
                
            df = pd.read_csv(file_path)
            self.min_vals = df.min()
            self.max_vals = df.max()
            
            # 归一化数据
            normalized_data = (df - self.min_vals) / (self.max_vals - self.min_vals)
            normalized_data = normalized_data * 1.8 - 0.9
            
            self.data = torch.FloatTensor(normalized_data.values)
            self.labels = torch.LongTensor([target_label] * len(df))
        else:
            # 加载所有数据
            for label, filename in file_mapping.items():
                file_path = os.path.join(csv_folder, filename)
                if not os.path.exists(file_path):
                    print(f"警告: 找不到文件 {filename}")
                    continue
                    
                df = pd.read_csv(file_path)
                
                if self.min_vals is None:
                    self.min_vals = df.min()
                    self.max_vals = df.max()
                
                normalized_data = (df - self.min_vals) / (self.max_vals - self.min_vals)
                normalized_data = normalized_data * 1.8 - 0.9
                
                self.data.append(torch.FloatTensor(normalized_data.values))
                self.labels.extend([label] * len(df))
            
            self.data = torch.cat(self.data, dim=0)
            self.labels = torch.LongTensor(self.labels)
        
        self.num_classes = len(file_mapping)
        
        print(f"加载数据: 形状={self.data.shape}, 类别数={self.num_classes}")
        if target_label is not None:
            print(f"目标标签={target_label}, 文件={file_mapping[target_label]}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),  # 使用BatchNorm
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1)  # 降低Dropout率
            ]
            return layers
        
        # 增加网络容量，与DCGAN保持一致
        self.model = nn.Sequential(
            *block(latent_dim * 2, 1024),
            *block(1024, 512),
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, output_dim),
            nn.Tanh()  # 使用Tanh
        )
        
    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, input_dim)
        
        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ]
            return layers
        
        self.model = nn.Sequential(
            *block(input_dim * 2, 1024),
            *block(1024, 512),
            *block(512, 256),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)

def train_lsgan(csv_folder, target_label, n_epochs=500, batch_size=64, lr=0.0002, latent_dim=100, beta1=0.5, beta2=0.999):
    """改进的LSGAN训练函数"""
    dataset = CSVDataset(csv_folder, target_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    feature_dim = dataset.data.shape[1]
    num_classes = dataset.num_classes
    
    generator = Generator(latent_dim, feature_dim, num_classes).cuda()
    discriminator = Discriminator(feature_dim, num_classes).cuda()
    
    # 使用MSE损失函数
    criterion = nn.MSELoss()
    
    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # 添加学习率调度器
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, n_epochs, eta_min=lr*0.1)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, n_epochs, eta_min=lr*0.1)
    
    best_g_loss = float('inf')
    best_generator_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(n_epochs):
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0
        
        for i, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            # 添加此检查，跳过太小的批次
            if batch_size <= 1:
                print(f"跳过批次 {i}，批次大小太小: {batch_size}")
                continue
                
            real_data = real_data.cuda()
            labels = labels.cuda()
            
            # 标签平滑
            real_target = torch.full((batch_size, 1), 0.9, dtype=torch.float).cuda()
            fake_target = torch.full((batch_size, 1), 0.1, dtype=torch.float).cuda()
            
            # 训练判别器
            d_optimizer.zero_grad()
            
            real_pred = discriminator(real_data, labels)
            d_real_loss = criterion(real_pred, real_target)
            
            z = torch.randn(batch_size, latent_dim).cuda()
            fake_data = generator(z, labels)
            fake_pred = discriminator(fake_data.detach(), labels)
            d_fake_loss = criterion(fake_pred, fake_target)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            
            fake_pred = discriminator(fake_data, labels)
            
            # 添加多样性损失
            diversity_loss = -torch.pdist(fake_data).mean()
            
            # 特征相关性损失
            correlation_loss = torch.abs(torch.corrcoef(fake_data.T)[0,1])
            
            # 添加分布匹配损失
            real_moments = torch.cat([real_data.mean(dim=0), real_data.std(dim=0)])
            fake_moments = torch.cat([fake_data.mean(dim=0), fake_data.std(dim=0)])
            moment_loss = F.mse_loss(fake_moments, real_moments)
            
            # 综合损失
            g_loss = (criterion(fake_pred, real_target) + 
                     0.1 * diversity_loss + 
                     0.1 * correlation_loss + 
                     0.2 * moment_loss)
            
            g_loss.backward()
            g_optimizer.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1
            
        # 更新学习率
        g_scheduler.step()
        d_scheduler.step()
        
        # 每10轮打印一次状态
        if epoch % 10 == 0:
            print(f"\nEpoch [{epoch}/{n_epochs}]")
            print(f"D Loss: {total_d_loss/num_batches:.4f}")
            print(f"G Loss: {total_g_loss/num_batches:.4f}")
            
            # 检查生成样本
            with torch.no_grad():
                test_z = torch.randn(100, latent_dim).cuda()
                test_labels = torch.LongTensor([target_label] * 100).cuda()
                test_samples = generator(test_z, test_labels).cpu().numpy()
                print("\n生成样本统计：")
                print(f"均值: {np.mean(test_samples, axis=0)}")
                print(f"标准差: {np.std(test_samples, axis=0)}")
                print(f"最小值: {np.min(test_samples, axis=0)}")
                print(f"最大值: {np.max(test_samples, axis=0)}")
        
        # 保存最佳模型
        if total_g_loss / num_batches < best_g_loss:
            best_g_loss = total_g_loss / num_batches
            best_generator_state = generator.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("\n早停：生成器损失没有改善")
            break
    
    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
    
    # 返回生成器和统计信息
    return {
        'generator_state_dict': generator.state_dict(),
        'min_vals': dataset.min_vals.values,
        'max_vals': dataset.max_vals.values,
    }

if __name__ == "__main__":
    csv_folder = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\features\BSD"
    
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
    
    # 创建保存目录
    save_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\gan_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个故障类型训练模型
    for label, fault_name in fault_types.items():
        print(f"\n开始训练 {fault_name} 故障的模型...")
        
        save_dict = train_lsgan(
        csv_folder=csv_folder,
        target_label=label,
        n_epochs=10,
        batch_size=64,
        lr=0.0002,
        beta1=0.5,  # 修正参数名为beta1
        beta2=0.999)
        
        # 使用故障名称保存模型
        save_path = os.path.join(save_dir, f"lsgan_generator_{fault_name}.pth")
        torch.save(save_dict, save_path)
        print(f"{fault_name} 故障的模型已保存到: {save_path}")
        print("-" * 50)
