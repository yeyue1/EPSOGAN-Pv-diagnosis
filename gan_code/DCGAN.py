import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import os

class CSVDataset(Dataset):
    def __init__(self, csv_folder, target_label=None):
        self.data = []
        self.labels = []
        self.min_vals = None
        self.max_vals = None
        
        # 统一的故障类型与文件名映射
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
            
            # 修改归一化方式，与LSGAN保持一致
            normalized_data = (df - self.min_vals) / (self.max_vals - self.min_vals)
            normalized_data = normalized_data * 1.8 - 0.9  # 归一化到[-0.9, 0.9]范围
            
            self.data = torch.FloatTensor(normalized_data.values)
            self.labels = torch.LongTensor([target_label] * len(df))
            
            print(f"加载数据文件: {file_mapping[target_label]}")
            print(f"数据形状: {self.data.shape}")
            
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
                normalized_data = normalized_data * 1.8 - 0.9  # 归一化到[-0.9, 0.9]范围
                
                self.data.append(torch.FloatTensor(normalized_data.values))
                self.labels.extend([label] * len(df))
            
            self.data = torch.cat(self.data, dim=0)
            self.labels = torch.LongTensor(self.labels)
        
        self.num_classes = len(file_mapping)
        
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
                nn.BatchNorm1d(out_feat),  # 使用BatchNorm替代LayerNorm
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1)  # 降低Dropout率
            ]
            return layers
            
        self.model = nn.Sequential(
            *block(latent_dim * 2, 1024),  # 增加网络容量
            *block(1024, 512),
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, input_dim)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),  # 增加网络容量
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),  # 降低Dropout率
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)

def train_dcgan(csv_folder, target_label, n_epochs=500, batch_size=64, lr=0.0002, 
                latent_dim=100, beta1=0.5, beta2=0.999):
    """改进的DCGAN训练函数"""
    dataset = CSVDataset(csv_folder, target_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    feature_dim = dataset.data.shape[1]
    num_classes = dataset.num_classes
    
    generator = Generator(latent_dim, feature_dim, num_classes).cuda()
    discriminator = Discriminator(feature_dim, num_classes).cuda()
    
    # 使用Adam优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # BCE损失函数
    criterion = nn.BCELoss()
    
    # 添加学习率调度器
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, n_epochs, eta_min=lr*0.1)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, n_epochs, eta_min=lr*0.1)
    
    real_label = 1
    fake_label = 0
    
    for epoch in range(n_epochs):
        for i, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.cuda()
            labels = labels.cuda()
            
            # 训练判别器
            d_optimizer.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float).cuda()
            
            # 真实数据的判别结果
            real_output = discriminator(real_data, labels).view(-1)
            d_loss_real = criterion(real_output, label)
            d_loss_real.backward()
            
            # 生成假数据
            noise = torch.randn(batch_size, latent_dim).cuda()
            fake_data = generator(noise, labels)
            label.fill_(fake_label)
            
            # 假数据的判别结果
            fake_output = discriminator(fake_data.detach(), labels).view(-1)
            d_loss_fake = criterion(fake_output, label)
            d_loss_fake.backward()
            
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            label.fill_(real_label)
            fake_output = discriminator(fake_data, labels).view(-1)
            g_loss = criterion(fake_output, label)
            g_loss.backward()
            g_optimizer.step()
            
            # 每100个batch打印一次状态
            if i % 100 == 0:
                print(f'[{epoch}/{n_epochs}][{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} '
                      f'D(x): {real_output.mean().item():.4f} D(G(z)): {fake_output.mean().item():.4f}')
                
                # 检查生成样本的范围
                with torch.no_grad():
                    test_noise = torch.randn(10, latent_dim).cuda()
                    test_labels = torch.LongTensor([target_label] * 10).cuda()
                    fake_samples = generator(test_noise, test_labels).cpu().numpy()
                    print(f"生成样本统计:")
                    print(f"Min: {fake_samples.min():.4f}, Max: {fake_samples.max():.4f}")
                    print(f"Mean: {fake_samples.mean():.4f}, Std: {fake_samples.std():.4f}")
        
        # 更新学习率
        g_scheduler.step()
        d_scheduler.step()
    
    # 返回生成器和统计信息
    return {
        'generator_state_dict': generator.state_dict(),
        'min_vals': dataset.min_vals.values,
        'max_vals': dataset.max_vals.values
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
        
        save_dict = train_dcgan(
        csv_folder=csv_folder,
        target_label=label,
        n_epochs=15,      # 增加训练轮数
        batch_size=64,     # 减小batch size
        lr=0.0002,        # 使用推荐的学习率
        latent_dim=100,
        beta1=0.5,
        beta2=0.999
    )
        
        # 使用故障类型名称保存模型
        save_path = os.path.join(save_dir, f"dcgan_generator_{fault_name}.pth")
        torch.save(save_dict, save_path)
        print(f"{fault_name} 故障的模型已保存到: {save_path}")
        print("-" * 50)