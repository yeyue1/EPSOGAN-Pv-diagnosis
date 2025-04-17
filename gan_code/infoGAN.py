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
            
            # 归一化数据
            normalized_data = (df - self.min_vals) / (self.max_vals - self.min_vals)
            normalized_data = normalized_data * 1.8 - 0.9
            
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
                normalized_data = normalized_data * 1.8 - 0.9
                
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
    def __init__(self, latent_dim, categorical_dim, continuous_dim, output_dim):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim
        
        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ]
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim + categorical_dim + continuous_dim, 1024),
            *block(1024, 512),
            *block(512, 256),
            nn.Linear(256, output_dim),
            nn.Hardtanh(min_val=-0.9, max_val=0.9)  # 修改输出激活函数范围
        )
    
    def forward(self, noise, categorical, continuous):
        gen_input = torch.cat((noise, categorical, continuous), dim=1)
        return self.model(gen_input)

class Discriminator(nn.Module):
    def __init__(self, input_dim, categorical_dim, continuous_dim):
        super(Discriminator, self).__init__()
        
        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ]
            return layers
        
        self.shared = nn.Sequential(
            *block(input_dim, 1024),
            *block(1024, 512),
            *block(512, 256)
        )
        
        # 真假判别器
        self.disc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 辅助分类器 (Q)
        self.categorical = nn.Linear(256, categorical_dim)
        self.continuous = nn.Linear(256, continuous_dim)
        
    def forward(self, x):
        features = self.shared(x)
        validity = self.disc(features)
        categorical = self.categorical(features)
        continuous = self.continuous(features)
        
        return validity, categorical, continuous

def train_infogan(csv_folder, target_label, n_epochs=50, batch_size=64, lr=0.0001, 
                 latent_dim=62, categorical_dim=7, continuous_dim=2, b1=0.5, b2=0.999):
    """为特定标签训练InfoGAN"""
    dataset = CSVDataset(csv_folder, target_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    feature_dim = dataset.data.shape[1]
    
    generator = Generator(latent_dim, categorical_dim, continuous_dim, feature_dim).cuda()
    discriminator = Discriminator(feature_dim, categorical_dim, continuous_dim).cuda()
    
    # 损失函数
    adversarial_loss = nn.BCELoss()
    categorical_loss = nn.CrossEntropyLoss()
    continuous_loss = nn.MSELoss()
    
    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    
    # 训练循环
    for epoch in range(n_epochs):
        for i, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            # 真实和虚假的标签
            valid = torch.ones(batch_size, 1).cuda()
            fake = torch.zeros(batch_size, 1).cuda()
            
            # 配置输入
            real_data = real_data.cuda()
            labels = labels.cuda()
            
            # -----------------
            # 训练判别器
            # -----------------
            d_optimizer.zero_grad()
            
            # 生成假数据
            z = torch.randn(batch_size, latent_dim).cuda()
            categorical_noise = torch.zeros(batch_size, categorical_dim).cuda()
            categorical_noise.scatter_(1, labels.unsqueeze(1), 1)
            continuous_noise = torch.randn(batch_size, continuous_dim).cuda()
            
            gen_data = generator(z, categorical_noise, continuous_noise)
            
            # 判别器输出
            real_validity, real_cat, real_cont = discriminator(real_data)
            fake_validity, fake_cat, fake_cont = discriminator(gen_data.detach())
            
            # 计算损失
            d_real_loss = adversarial_loss(real_validity, valid)
            d_fake_loss = adversarial_loss(fake_validity, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            d_optimizer.step()
            
            # -----------------
            # 训练生成器
            # -----------------
            g_optimizer.zero_grad()
            
            # 重新生成假数据
            gen_data = generator(z, categorical_noise, continuous_noise)
            fake_validity, fake_cat, fake_cont = discriminator(gen_data)
            
            # 计算信息损失
            info_categorical_loss = categorical_loss(fake_cat, labels)
            info_continuous_loss = continuous_loss(fake_cont, continuous_noise)
            
            # 计算总损失
            g_loss = adversarial_loss(fake_validity, valid) + \
                    info_categorical_loss + \
                    info_continuous_loss
            
            g_loss.backward()
            g_optimizer.step()
            
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{n_epochs}] "
                  f"D Loss: {d_loss.item():.4f} "
                  f"G Loss: {g_loss.item():.4f} "
                  f"Info Loss: {(info_categorical_loss + info_continuous_loss).item():.4f}")
            
            # 生成一些样本进行检查
            with torch.no_grad():
                z = torch.randn(10, latent_dim).cuda()
                categorical = torch.zeros(10, categorical_dim).cuda()
                categorical[:, target_label] = 1
                continuous = torch.randn(10, continuous_dim).cuda()
                fake_data = generator(z, categorical, continuous).cpu().numpy()
                print("生成样本范围：")
                print(f"Min: {fake_data.min(axis=0)}")
                print(f"Max: {fake_data.max(axis=0)}")
    
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
        
        save_dict = train_infogan(
            csv_folder=csv_folder,
            target_label=label,
            n_epochs=30,
            batch_size=64
        )
        
        # 使用故障名称保存模型
        save_path = os.path.join(save_dir, f"infogan_generator_{fault_name}.pth")
        torch.save(save_dict, save_path)
        print(f"{fault_name} 故障的模型已保存到: {save_path}")
        print("-" * 50)