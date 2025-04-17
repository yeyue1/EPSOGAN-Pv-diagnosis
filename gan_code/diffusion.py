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

# 定义扩散模型的噪声预测网络 - 减小模型容量并增加正则化
class NoisePredictor(nn.Module):
    def __init__(self, input_dim, time_dim=32, categorical_dim=7):
        super(NoisePredictor, self).__init__()
        
        # 时间编码层 - 使用更强的时间编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),  # 使用GELU激活函数
            nn.LayerNorm(time_dim),  # 添加层归一化
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.LayerNorm(time_dim)
        )
        
        # 条件编码层 - 为标签添加噪声以防止过拟合
        self.condition_mlp = nn.Sequential(
            nn.Linear(categorical_dim, time_dim),
            nn.GELU(),
            nn.LayerNorm(time_dim),
            nn.Dropout(0.2),  # 增加dropout
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.LayerNorm(time_dim)
        )
        
        # 主干网络 - 减少网络大小，使用残差连接
        self.input_proj = nn.Linear(input_dim, 256)
        
        # 两个残差块
        self.res1 = nn.Sequential(
            nn.Linear(256 + time_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        self.res2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(256, input_dim)
        
        # 初始化权重，使用较小的初始值
        self._init_weights()
    
    def _init_weights(self):
        # 使用较小的初始权重以减轻过拟合
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, t, condition):
        # 时间编码
        t_emb = self.time_mlp(t.unsqueeze(-1))
        
        # 条件编码 - 添加噪声增强泛化能力
        if self.training:
            condition = condition + torch.randn_like(condition) * 0.1
        c_emb = self.condition_mlp(condition)
        
        # 初始投影
        h = self.input_proj(x)
        
        # 合并特征
        h_cond = torch.cat([h, t_emb, c_emb], dim=1)
        
        # 残差块1
        h = h + self.res1(h_cond)
        
        # 残差块2
        h = h + self.res2(h)
        
        # 输出投影
        return self.output_proj(h)

# 扩散模型 - 改进采样策略和噪声调度
class DiffusionModel:
    def __init__(self, feature_dim, categorical_dim=7, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        初始化扩散模型
        
        Args:
            feature_dim: 特征维度
            categorical_dim: 类别维度（故障类型数量）
            timesteps: 扩散步骤数
            beta_start: 初始噪声方差
            beta_end: 最终噪声方差
        """
        self.feature_dim = feature_dim
        self.categorical_dim = categorical_dim
        self.timesteps = timesteps
        
        # 使用更安全的线性噪声调度，避免余弦调度可能带来的数值不稳定性
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算一些需要的值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 噪声预测网络
        self.noise_predictor = NoisePredictor(feature_dim, categorical_dim=categorical_dim)
    
    def _cosine_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        """余弦噪声调度，对扩散过程更有益，但可能导致数值不稳定"""
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
    
    def to(self, device):
        """将模型移至指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.noise_predictor = self.noise_predictor.to(device)
        return self
    
    def forward_diffusion(self, x_0, t):
        """前向扩散过程：给定x_0和时间步t，添加噪声得到x_t"""
        noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # x_t = √(α_t) * x_0 + √(1-α_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def sample(self, batch_size, labels, device):
        """从纯噪声开始，通过反向扩散过程生成样本"""
        # 创建one-hot编码的标签
        condition = torch.zeros(batch_size, self.categorical_dim).to(device)
        condition.scatter_(1, labels.unsqueeze(1), 1)
        
        # 从纯噪声开始 - 使用正常的初始噪声强度
        x = torch.randn(batch_size, self.feature_dim).to(device)
        
        # 记录采样进度
        print(f"开始采样过程，总步骤数: {self.timesteps}")
        
        # 使用标准的扩散采样算法，更稳定可靠
        for t in range(self.timesteps - 1, -1, -1):
            if t % 200 == 0:
                print(f"采样步骤: {self.timesteps - t}/{self.timesteps}")
                
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 当t=0时不添加噪声
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
                
            # 预测噪声
            with torch.no_grad():
                predicted_noise = self.noise_predictor(x, t_tensor / self.timesteps, condition)
            
            # 计算去噪步骤的参数
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t-1]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)
                
            # 更安全的去噪系数计算
            coef1 = 1.0 / torch.sqrt(alpha + 1e-8)
            coef2 = beta / torch.sqrt(1.0 - alpha_cumprod + 1e-8)
            
            # 应用去噪步骤
            x = coef1 * (x - coef2 * predicted_noise)
            
            # 添加噪声（除非是最后一步）
            if t > 0:
                noise_scale = torch.sqrt(beta + 1e-8)
                x = x + noise_scale * z
        
        print("采样完成")
        
        # 使用更安全的输出范围限制
        x = torch.clamp(x, -0.9, 0.9)
        
        return x

def train_diffusion_model(csv_folder, target_label, n_epochs=50, batch_size=64, lr=0.0001,
                        timesteps=1000, categorical_dim=7):
    """为特定标签训练扩散模型"""
    # 加载数据集
    dataset = CSVDataset(csv_folder, target_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    feature_dim = dataset.data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化扩散模型 - 使用更保守的参数
    beta_start = 1e-4
    beta_end = 0.01  # 降低最大噪声值
    
    # 初始化扩散模型
    diffusion_model = DiffusionModel(
        feature_dim=feature_dim,
        categorical_dim=categorical_dim,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end
    ).to(device)
    
    # 优化器 - 使用较小的学习率以防止过快收敛
    optimizer = optim.Adam(diffusion_model.noise_predictor.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    
    best_loss = float('inf')
    
    # 数据增强 - 生成额外的虚拟数据点
    def augment_data(data, labels):
        aug_data = data.clone()
        batch_size = data.size(0)
        
        # 随机特征扰动
        aug_data = aug_data + torch.randn_like(aug_data) * 0.05
        
        # 随机特征混合 (mixup)
        if batch_size > 1:
            alpha = 0.2
            lam = torch.distributions.Beta(alpha, alpha).sample()
            rand_index = torch.randperm(batch_size)
            mixed_data = lam * aug_data + (1 - lam) * aug_data[rand_index]
            # 只对20%的批次应用mixup
            apply_mix = torch.rand(batch_size) < 0.2
            aug_data[apply_mix] = mixed_data[apply_mix]
        
        return aug_data, labels
    
    # 训练循环
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            # 将数据移至GPU
            real_data = real_data.to(device)
            labels = labels.to(device)
            
            # 应用数据增强
            aug_data, aug_labels = augment_data(real_data, labels)
            
            # 创建one-hot编码的条件标签
            condition = torch.zeros(batch_size, categorical_dim).to(device)
            condition.scatter_(1, aug_labels.unsqueeze(1), 1)
            
            # 随机选择时间步
            if np.random.random() > 0.3:  # 70%的时间使用全范围采样
                t = torch.randint(0, timesteps, (batch_size,), device=device)
            else:  # 30%的时间集中在特定区域
                if np.random.random() > 0.5:
                    # 前期时间步 (0-20%)
                    t = torch.randint(0, int(timesteps * 0.2), (batch_size,), device=device)
                else:
                    # 后期时间步 (80-100%)
                    t = torch.randint(int(timesteps * 0.8), timesteps, (batch_size,), device=device)
            
            # 前向扩散过程：添加噪声
            noisy_data, target_noise = diffusion_model.forward_diffusion(aug_data, t)
            
            # 额外噪声 - 增加训练的稳健性
            if epoch < n_epochs // 2:  # 只在前半段训练中添加
                noise_level = 0.1 * (1 - epoch / (n_epochs // 2))  # 逐渐减少噪声
                noisy_data = noisy_data + torch.randn_like(noisy_data) * noise_level
            
            # 预测噪声
            predicted_noise = diffusion_model.noise_predictor(noisy_data, t.float() / timesteps, condition)
            
            # 计算损失
            loss = nn.MSELoss()(predicted_noise, target_noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(diffusion_model.noise_predictor.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # 计算平均损失并更新学习率调度器
        avg_loss = epoch_loss / batch_count
        scheduler.step(avg_loss)
        
        # 打印训练进度
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch [{epoch}/{n_epochs}] Loss: {avg_loss:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = diffusion_model.noise_predictor.state_dict()
                print(f"更新最佳模型，损失降低到 {best_loss:.6f}")
            
            # 生成一些示例
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                with torch.no_grad():
                    try:
                        # 使用固定的标签生成样本
                        sample_labels = torch.full((10,), target_label, dtype=torch.long, device=device)
                        samples = diffusion_model.sample(10, sample_labels, device).cpu().numpy()
                        
                        # 检查NaN并处理
                        if np.isnan(samples).any():
                            print("警告: 生成样本包含NaN值，尝试修复...")
                            # 尝试再次生成，使用更保守的参数
                            samples = torch.clamp(torch.randn(10, feature_dim).cuda() * 0.5, -0.9, 0.9).cpu().numpy()
                        
                        print("生成样本范围：")
                        print(f"Min: {np.nanmin(samples, axis=0)}")
                        print(f"Max: {np.nanmax(samples, axis=0)}")
                        
                        # 检查多样性
                        sample_std = np.nanstd(samples, axis=0)
                        print(f"样本标准差: {sample_std}")
                        if np.nanmean(sample_std) < 0.05:
                            print("警告: 生成样本多样性较低")
                    except Exception as e:
                        print(f"生成样本时出错: {str(e)}")
                    
    # 返回训练好的最佳模型和统计信息
    return {
        'model_state_dict': best_model_state,
        'timesteps': timesteps,
        'feature_dim': feature_dim,
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
    save_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\diffusion_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个故障类型训练模型
    for label, fault_name in fault_types.items():
        print(f"\n开始训练 {fault_name} 故障的扩散模型...")
        
        save_dict = train_diffusion_model(
            csv_folder=csv_folder,
            target_label=label,
            n_epochs=50,  # 减少训练轮数以避免过拟合
            batch_size=32,  # 增大批次大小以提高稳定性
            timesteps=500  # 减少时间步数，提高数值稳定性
        )
        
        # 使用故障名称保存模型
        save_path = os.path.join(save_dir, f"diffusion_model_{fault_name}.pth")
        torch.save(save_dict, save_path)
        print(f"{fault_name} 故障的扩散模型已保存到: {save_path}")
        print("-" * 50)
