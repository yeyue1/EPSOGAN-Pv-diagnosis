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

# 修改Generator类的结构
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        def block(in_feat, out_feat):
            layers = [
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1)
            ]
            return layers
            
        self.model = nn.Sequential(
            *block(latent_dim * 2, 1024),  # 与DCGAN保持一致的网络结构
            *block(1024, 512),
            *block(512, 256),
            *block(256, 128),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, input_dim)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)

# 在train_wgan_gp函数中修改参数
def train_wgan_gp(csv_folder, target_label, n_epochs=500, batch_size=64, lr=0.0002, 
                  latent_dim=100, lambda_gp=10):
    # 加载特定标签的数据
    dataset = CSVDataset(csv_folder, target_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 获取特征维度
    feature_dim = dataset.data.shape[1]
    
    # 初始化模型
    generator = Generator(latent_dim, feature_dim, dataset.num_classes).cuda()
    discriminator = Discriminator(feature_dim, dataset.num_classes).cuda()
    
    # 修改优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 学习率调度器
    g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        g_optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        d_optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    best_g_loss = float('inf')
    best_generator_state = None
    patience = 50
    patience_counter = 0
    
    # 训练循环
    for epoch in range(n_epochs):
        total_g_loss = 0
        total_d_loss = 0
        num_batches = 0
        
        for i, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.cuda()
            labels = labels.cuda()
            
            # 训练判别器
            for _ in range(5):  # critic iterations
                d_optimizer.zero_grad()
                
                z = torch.randn(batch_size, latent_dim).cuda()
                fake_data = generator(z, labels)
                
                # 添加噪声到真实和生成的数据
                real_data_noisy = real_data + 0.05 * torch.randn_like(real_data)
                fake_data_noisy = fake_data + 0.05 * torch.randn_like(fake_data)
                
                real_validity = discriminator(real_data_noisy, labels)
                fake_validity = discriminator(fake_data_noisy.detach(), labels)
                
                # 计算梯度惩罚
                alpha = torch.rand(batch_size, 1).cuda()
                interpolates = (alpha * real_data + (1 - alpha) * fake_data.detach()).requires_grad_(True)
                d_interpolates = discriminator(interpolates, labels)
                
                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates).cuda(),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                d_optimizer.step()
            
            # 训练生成器（多次更新）
            for _ in range(2):  # 增加生成器的训练次数
                g_optimizer.zero_grad()
                fake_data = generator(z, labels)
                fake_validity = discriminator(fake_data, labels)
                
                # 添加多样性损失
                diversity_loss = -torch.pdist(fake_data).mean()
                g_loss = -torch.mean(fake_validity) + 0.1 * diversity_loss
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                g_optimizer.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        
        # 更新学习率
        g_scheduler.step(avg_g_loss)
        d_scheduler.step(avg_d_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{n_epochs}] "
                  f"D Loss: {avg_d_loss:.4f} G Loss: {avg_g_loss:.4f}")
            
            # 修改生成样本检查的代码
            with torch.no_grad():
                n_test = 10  # 测试样本数量
                z = torch.randn(n_test, latent_dim).cuda()
                test_labels = torch.LongTensor([target_label] * n_test).cuda()  # 使用固定标签
                fake_data = generator(z, test_labels).cpu().numpy()
                
                # 反归一化
                fake_data = (fake_data + 0.9) / 1.8
                fake_data = fake_data * (dataset.max_vals.values - dataset.min_vals.values) + dataset.min_vals.values
                print("生成样本范围：")
                print(f"Min: {fake_data.min(axis=0)}")
                print(f"Max: {fake_data.max(axis=0)}")
        
        # 保存最佳模型
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_generator_state = generator.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print("早停：生成器损失没有改善")
            break
    
    # 使用最佳生成器状态
    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
    
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
        
        # 训练模型并获取保存字典
        # 在main函数中修改参数
        save_dict = train_wgan_gp(
            csv_folder=csv_folder,
            target_label=label,
            n_epochs=10,      # 增加训练轮数
            batch_size=64,     # 与DCGAN保持一致
            lr=0.001,
            latent_dim=100,
            lambda_gp=10
        )
        
        # 使用故障名称保存模型
        save_path = os.path.join(save_dir, f"wgan_gp_generator_{fault_name}.pth")
        torch.save(save_dict, save_path)
        print(f"{fault_name} 故障的模型已保存到: {save_path}")
        print("-" * 50)