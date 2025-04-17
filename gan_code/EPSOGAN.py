import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import os

fault_types = [
        'aging', 'normal', 'open', 'shading', 
        'short', 'aging_open', 'aging_short'
    ]

def MMD(G, X_real=None, kernel='rbf', sigma=1.0):
    """最大均值差异(Maximum Mean Discrepancy)
    Args:
        G: 生成的样本
        X_real: 真实样本
        kernel: 核函数类型
        sigma: RBF核的带宽参数
    """
    if X_real is None:
        # 如果没有提供真实样本，使用随机生成的样本作为示例
        X_real = np.random.randn(*G.shape)
    
    def rbf_kernel(X, Y):
        distances = cdist(X, Y, 'sqeuclidean')
        return np.exp(-distances / (2 * sigma ** 2))
    
    K_XX = rbf_kernel(X_real, X_real)
    K_YY = rbf_kernel(G, G)
    K_XY = rbf_kernel(X_real, G)
    
    m = X_real.shape[0]
    n = G.shape[0]
    
    mmd = (np.sum(K_XX) / (m * m) + 
           np.sum(K_YY) / (n * n) - 
           2 * np.sum(K_XY) / (m * n))
    
    return np.sqrt(max(mmd, 0))

def Coverage(G, X_real=None, n_clusters=10):
    """覆盖率评估 - 基于生成样本对真实数据分布的覆盖程度
    Args:
        G: 生成的样本 (numpy.ndarray)
        X_real: 真实样本 (numpy.ndarray)
        n_clusters: 聚类数量
    Returns:
        coverage: 覆盖率得分 (0-1之间的浮点数)
    """
    if X_real is None:
        X_real = np.random.randn(*G.shape)
    
    try:
        # 设置OpenMP线程数以避免内存泄漏
        os.environ['OMP_NUM_THREADS'] = '4'
        
        # 确保数据格式正确
        G = np.array(G, dtype=np.float32)
        X_real = np.array(X_real, dtype=np.float32)
        
        # 1. 对真实数据进行聚类，显式设置n_init参数
        kmeans_real = KMeans(
            n_clusters=n_clusters, 
            random_state=42,
            n_init=10  # 显式设置n_init值
        ).fit(X_real)
        real_centers = kmeans_real.cluster_centers_
        
        # 2. 计算每个生成样本到最近真实聚类中心的距离
        distances_to_centers = cdist(G, real_centers)
        min_distances = np.min(distances_to_centers, axis=1)
        
        # 3. 计算真实数据的分布特征
        real_distances = cdist(X_real, real_centers)
        real_min_distances = np.min(real_distances, axis=1)
        distance_threshold = np.percentile(real_min_distances, 75)
        
        # 4. 计算每个聚类中心是否被覆盖
        covered_centers = set()
        for i, gen_sample in enumerate(G):
            if min_distances[i] <= distance_threshold:
                nearest_center = np.argmin(distances_to_centers[i])
                covered_centers.add(nearest_center)
        
        # 5. 计算覆盖率
        coverage = len(covered_centers) / n_clusters
        
        # 调试信息
        print(f"\nCoverage统计信息:")
        print(f"总聚类中心数: {n_clusters}")
        print(f"被覆盖的聚类中心数: {len(covered_centers)}")
        print(f"距离阈值: {distance_threshold:.4f}")
        print(f"覆盖率: {coverage:.4f}")
        
        return coverage
        
    except Exception as e:
        print(f"计算Coverage时出错: {str(e)}")
        return 0.5  # 发生错误时返回中性值
    finally:
        # 清理环境变量
        if 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']

def Wasserstein(G, X_real=None):
    """Wasserstein距离计算
    Args:
        G: 生成的样本
        X_real: 真实样本
    """
    if X_real is None:
        X_real = np.random.randn(*G.shape)
    
    # 计算每个维度的Wasserstein距离
    wasserstein_distances = []
    for i in range(G.shape[1]):
        dist = wasserstein_distance(G[:, i], X_real[:, i])
        wasserstein_distances.append(dist)
    
    # 返回平均Wasserstein距离
    return np.mean(wasserstein_distances)

def ModeScore(G, X_real=None, n_clusters=10):
    """计算标准的Mode Score
    Args:
        G: 生成的样本
        X_real: 真实样本
        n_clusters: 聚类数量用于近似条件分布
    Returns:
        mode_score: Mode Score值
    """
    if X_real is None:
        X_real = np.random.randn(*G.shape)
        
    try:
        # 1. 对真实数据和生成数据进行聚类
        kmeans_real = KMeans(n_clusters=n_clusters, random_state=42).fit(X_real)
        kmeans_gen = KMeans(n_clusters=n_clusters, random_state=42).fit(G)
        
        # 2. 计算条件分布 p(y|x)
        real_labels = kmeans_real.predict(X_real)
        gen_labels = kmeans_real.predict(G)  # 使用真实数据的聚类中心
        
        # 计算真实数据的类别分布 p(y_train)
        p_y_train = np.bincount(real_labels, minlength=n_clusters) / len(real_labels)
        
        # 计算生成数据的类别分布 p(y)
        p_y = np.bincount(gen_labels, minlength=n_clusters) / len(gen_labels)
        
        # 3. 计算条件KL散度 E[KL(p(y|x) || p(y_train))]
        kl_conditional = 0
        for i in range(n_clusters):
            # 计算每个聚类的条件概率
            p_y_given_x = np.zeros(n_clusters)
            mask = (gen_labels == i)
            if np.any(mask):
                cluster_labels = kmeans_real.predict(G[mask])
                p_y_given_x = np.bincount(cluster_labels, minlength=n_clusters) / len(cluster_labels)
                
            # 避免0值
            p_y_given_x = np.clip(p_y_given_x, 1e-10, 1.0)
            p_y_train_clip = np.clip(p_y_train, 1e-10, 1.0)
            
            # 计算KL散度
            kl = np.sum(p_y_given_x * np.log(p_y_given_x / p_y_train_clip))
            kl_conditional += kl * (np.sum(mask) / len(G))
        
        # 4. 计算边缘KL散度 KL(p(y) || p(y_train))
        p_y = np.clip(p_y, 1e-10, 1.0)
        p_y_train = np.clip(p_y_train, 1e-10, 1.0)
        kl_marginal = np.sum(p_y * np.log(p_y / p_y_train))
        
        # 5. 计算最终的Mode Score
        mode_score = np.exp(kl_conditional - kl_marginal)
        
        # 打印调试信息
        print(f"\nMode Score 计算详情:")
        print(f"条件KL散度: {kl_conditional:.4f}")
        print(f"边缘KL散度: {kl_marginal:.4f}")
        print(f"Mode Score: {mode_score:.4f}")
        
        return mode_score
        
    except Exception as e:
        print(f"计算Mode Score时出错: {str(e)}")
        return float('inf')

def f_GCaculate(mmd_score, coverage_score, wasserstein_score, mode_score):
    """计算综合评价指标
    Args:
        mmd_score: MMD评分
        coverage_score: Coverage评分
        wasserstein_score: Wasserstein评分
        mode_score: Mode评分
    Returns:
        f_G: 综合评价指标值
    """
    # 计算综合得分（负号表示我们要最小化这个值）
    f_G = -((mmd_score + 1) * (coverage_score + 1) * (wasserstein_score + 1) * (mode_score + 1))
    return f_G

def calculate_all_f_G():
    """计算所有GAN的F_G值"""
    try:
        # 读取评估结果
        results_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\gan_evaluation_results.csv"
        results_df = pd.read_csv(results_path, index_col=0)
        
        # 存储每个GAN的F_G值
        f_g_scores = {}
        
        # 为每个GAN计算F_G
        for gan_type in results_df.index:
            scores = results_df.loc[gan_type]
            f_g = f_GCaculate(
                scores['MMD'],
                scores['Coverage'],
                scores['Wasserstein'],
                scores['ModeScore']
            )
            f_g_scores[gan_type] = f_g
        
        # 将结果添加到原始数据框中
        results_df['F_G'] = pd.Series(f_g_scores)
        
        # 保存更新后的结果
        results_df.to_csv(results_path)
        
        # 打印结果
        print("\nF_G计算结果:")
        for gan_type, score in f_g_scores.items():
            print(f"{gan_type}: {score:.4f}")
        
        # 找出最佳GAN（F_G值最大的）
        best_gan = max(f_g_scores.items(), key=lambda x: x[1])
        print(f"\n最佳GAN (基于F_G): {best_gan[0]} (F_G = {best_gan[1]:.4f})")
        
        return f_g_scores
        
    except Exception as e:
        print(f"计算F_G时出错: {str(e)}")
        return None

def objective_function(pso_variables):
    """计算目标函数：所有GAN的F_G值与PSO变量的加权和
    Args:
        pso_variables: PSO算法生成的权重变量数组 (5个变量)
    Returns:
        weighted_sum: 加权总和
    """
    try:
        # 定义默认的F_G值（基于已知结果）
        default_f_g_values = {
            'CTGAN': -1149.4190788927442,
            'DCGAN': -129.3108552534704,
            'infoGAN': -892.3393004819247,
            'LSGAN': -776.3451757204983,
            'WGAN_GP': -1608.9807426557593
        }
        
        # 获取每个GAN的F_G值
        gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
        f_g_values = [default_f_g_values[gan_type] for gan_type in gan_types]
        
        # 计算加权和
        weighted_sum = sum(f_g * w for f_g, w in zip(f_g_values, pso_variables))
        return weighted_sum
        
    except Exception as e:
        print(f"计算目标函数时出错: {str(e)}")
        return float('inf')  # 返回无穷大表示出错

def objective_functions(pso_variables):
    """多个目标函数的计算"""
    try:
        # 修改为读取evaluation_summary.csv
        eval_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\evaluation_results"
        summary_file = os.path.join(eval_dir, 'evaluation_summary.csv')
        
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"找不到评估总结文件: {summary_file}")
            
        results_df = pd.read_csv(summary_file, index_col=0)
        gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
        
        # 从总结文件中获取每个GAN的最佳指标值
        obj1 = sum(float(results_df.loc[gan_type, 'Best_MMD_Value']) * w 
                  for gan_type, w in zip(gan_types, pso_variables))
        obj2 = -sum(float(results_df.loc[gan_type, 'Best_Coverage_Value']) * w 
                    for gan_type, w in zip(gan_types, pso_variables))
        obj3 = sum(float(results_df.loc[gan_type, 'Best_Wasserstein_Value']) * w 
                  for gan_type, w in zip(gan_types, pso_variables))
        obj4 = -sum(float(results_df.loc[gan_type, 'Best_ModeScore_Value']) * w 
                    for gan_type, w in zip(gan_types, pso_variables))
        
        return np.array([obj1, obj2, obj3, obj4])
        
    except Exception as e:
        #print(f"计算多目标函数时出错: {str(e)}")
        return np.array([float('inf')] * 4)

class MOPSO:
    """多目标粒子群优化算法"""
    def __init__(self, num_particles, num_dimensions, num_objectives, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_objectives = num_objectives
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 初始化粒子位置和速度
        self.positions = np.random.rand(num_particles, num_dimensions)
        self.positions = self.normalize_positions(self.positions)
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
        
        # 初始化个体最优解和全局最优解
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.array([objective_functions(p) for p in self.positions])
        
        # 初始化帕累托最优解集
        self.pareto_front = []
        self.pareto_positions = []
        self.update_pareto_front()

    def normalize_positions(self, positions):
        """确保位置非负且和为1"""
        positions = np.abs(positions)
        row_sums = positions.sum(axis=1, keepdims=True)
        return positions / row_sums

    def dominates(self, scores1, scores2):
        """判断scores1是否支配scores2"""
        return np.all(scores1 <= scores2) and np.any(scores1 < scores2)

    def update_pareto_front(self):
        """更新帕累托前沿"""
        scores = [objective_functions(p) for p in self.positions]
        self.pareto_front = []
        self.pareto_positions = []
        
        for i, score in enumerate(scores):
            dominated = False
            for other_score in scores:
                if self.dominates(other_score, score):
                    dominated = True
                    break
            if not dominated:
                self.pareto_front.append(score)
                self.pareto_positions.append(self.positions[i])

    def select_leader(self):
        """选择全局最优位置"""
        if not self.pareto_positions:
            return self.positions[0]
        return self.pareto_positions[np.random.randint(len(self.pareto_positions))]

    def update_velocities(self):
        """更新粒子速度"""
        r1 = np.random.rand(self.num_particles, self.num_dimensions)
        r2 = np.random.rand(self.num_particles, self.num_dimensions)
        
        leader = self.select_leader()
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social = self.c2 * r2 * (leader - self.positions)
        
        self.velocities = self.w * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, -1, 1)

    def update_positions(self):
        """更新粒子位置"""
        self.positions += self.velocities
        self.positions = self.normalize_positions(self.positions)

    def optimize(self):
        """运行MOPSO算法"""
        for iter_num in range(self.max_iter):
            # 更新位置和速度
            self.update_velocities()
            self.update_positions()
            
            # 更新个体最优解
            current_scores = np.array([objective_functions(p) for p in self.positions])
            for i in range(self.num_particles):
                if self.dominates(current_scores[i], self.pbest_scores[i]):
                    self.pbest_positions[i] = self.positions[i]
                    self.pbest_scores[i] = current_scores[i]
            
            # 更新帕累托前沿
            self.update_pareto_front()
            
            # 每10次迭代打印一次
            if iter_num % 10 == 0:
                print(f"\n迭代 {iter_num}:")
                print(f"帕累托解数量: {len(self.pareto_front)}")
                print("部分帕累托最优解:")
                for pos, scores in zip(self.pareto_positions[:3], self.pareto_front[:3]):
                    print(f"权重: {pos}")
                    print(f"目标函数值: {scores}")
                    print(f"权重和: {np.sum(pos):.6f}")

        return self.pareto_positions, self.pareto_front

# PSO算法
class PSO:
    def __init__(self, num_particles, num_dimensions, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 初始化粒子的位置（确保和为1且都为正）
        self.positions = np.random.rand(num_particles, num_dimensions)
        self.positions = self.normalize_positions(self.positions)
        
        # 初始化速度（范围限制在[-1,1]）
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
        
        # 初始化个体最优解和全局最优解
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.array([objective_function(p) for p in self.positions])
        self.gbest_position = self.pbest_positions[np.argmin(self.pbest_scores)]
        self.gbest_score = np.min(self.pbest_scores)

    def normalize_positions(self, positions):
        """确保位置非负且和为1"""
        # 确保所有值非负
        positions = np.abs(positions)
        # 对每个粒子的维度进行归一化，使其和为1
        row_sums = positions.sum(axis=1, keepdims=True)
        normalized_positions = positions / row_sums
        return normalized_positions

    def update_velocities(self):
        """更新粒子的速度，并限制范围"""
        r1 = np.random.rand(self.num_particles, self.num_dimensions)
        r2 = np.random.rand(self.num_particles, self.num_dimensions)
        
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social = self.c2 * r2 * (self.gbest_position - self.positions)
        
        # 更新速度
        self.velocities = self.w * self.velocities + cognitive + social
        
        # 限制速度范围
        self.velocities = np.clip(self.velocities, -1, 1)

    def update_positions(self):
        """更新粒子的位置，确保非负且和为1"""
        # 更新位置
        self.positions += self.velocities
        
        # 确保位置非负且和为1
        self.positions = self.normalize_positions(self.positions)

    def optimize(self):
        """运行PSO算法"""
        for _ in range(self.max_iter):
            # 评估目标函数
            scores = np.array([objective_function(p) for p in self.positions])
            
            # 更新个体最优解
            better_mask = scores < self.pbest_scores
            self.pbest_positions[better_mask] = self.positions[better_mask]
            self.pbest_scores[better_mask] = scores[better_mask]

            # 更新全局最优解
            min_index = np.argmin(self.pbest_scores)
            if self.pbest_scores[min_index] < self.gbest_score:
                self.gbest_position = self.pbest_positions[min_index]
                self.gbest_score = self.pbest_scores[min_index]

            # 更新速度和位置
            self.update_velocities()
            self.update_positions()

            # 打印当前最优解的权重
            if _ % 10 == 0:  # 每10次迭代打印一次
                print(f"\n迭代 {_}:")
                print(f"当前最优权重: {self.gbest_position}")
                print(f"权重和: {np.sum(self.gbest_position):.6f}")
                print(f"目标函数值: {self.gbest_score:.6f}")

        return self.gbest_position, self.gbest_score

def save_results(pareto_positions, pareto_front, gan_types):
    """保存多目标优化结果到CSV文件
    Args:
        pareto_positions: 帕累托最优权重
        pareto_front: 对应的目标函数值
        gan_types: GAN类型列表
    """
    results_df = pd.DataFrame()
    
    # 添加每个解的权重
    for i, gan_type in enumerate(gan_types):
        results_df[f'Weight_{gan_type}'] = [pos[i] for pos in pareto_positions]
    
    # 添加目标函数值
    results_df['MMD'] = [front[0] for front in pareto_front]
    results_df['Coverage'] = [-front[1] for front in pareto_front]  # 转回原始值（取消负号）
    results_df['Wasserstein'] = [front[2] for front in pareto_front]
    results_df['ModeScore'] = [-front[3] for front in pareto_front]  # 转回原始值（取消负号）
    
    # 保存结果
    output_path = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

def objective_functions_by_fault(pso_variables, fault_type):
    """针对特定故障类型的多目标函数计算"""
    try:
        results_path = os.path.join(
            r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\evaluation_results",
            f'evaluation_results_{fault_type}.csv'
        )
        
        # 读取CSV文件，确保索引设置正确
        results_df = pd.read_csv(results_path)
        # 确保第一列作为索引
        if 'Unnamed: 0' in results_df.columns:
            results_df.set_index('Unnamed: 0', inplace=True)
        else:
            results_df.set_index(results_df.columns[0], inplace=True)
        
        gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
        
        # 检查数据是否存在
        for gan in gan_types:
            if gan not in results_df.index:
                print(f"警告: 在评估结果中找不到 {gan}")
                print("当前索引:", results_df.index.tolist())
                raise KeyError(f"找不到GAN类型: {gan}")
        
        # 计算四个目标函数
        objectives = []
        metrics = [('MMD', False), ('Coverage', True), 
                  ('Wasserstein', False), ('ModeScore', True)]
        
        for metric, reverse in metrics:
            if metric not in results_df.columns:
                raise KeyError(f"找不到指标: {metric}")
            
            values = []
            for gan, w in zip(gan_types, pso_variables):
                try:
                    value = float(results_df.loc[gan, metric]) * w
                    values.append(value)
                except Exception as e:
                    print(f"处理 {gan} 的 {metric} 时出错:")
                    print(f"值: {results_df.loc[gan, metric]}")
                    print(f"权重: {w}")
                    raise e
            
            total = sum(values)
            objectives.append(-total if reverse else total)
        
        return np.array(objectives)
        
    except Exception as e:
        print(f"处理 {fault_type} 故障时出错: {str(e)}")
        if 'results_df' in locals():
            print("\n数据内容:")
            print(results_df.head())
            print("\n索引:", results_df.index.tolist())
            print("列名:", results_df.columns.tolist())
        return np.array([float('inf')] * 4)

class MOPSOByFault(MOPSO):
    """针对特定故障类型的MOPSO"""
    def __init__(self, fault_type, num_particles, num_dimensions, num_objectives, max_iter, w=0.5, c1=1.5, c2=1.5):
        # 先设置故障类型
        self.fault_type = fault_type
        # 再调用父类初始化
        super().__init__(num_particles, num_dimensions, num_objectives, max_iter, w, c1, c2)
        
    def update_pareto_front(self):
        """更新帕累托前沿，使用特定故障类型的目标函数"""
        scores = [objective_functions_by_fault(p, self.fault_type) for p in self.positions]
        self.pareto_front = []
        self.pareto_positions = []
        
        for i, score in enumerate(scores):
            dominated = False
            for other_score in scores:
                if self.dominates(other_score, score):
                    dominated = True
                    break
            if not dominated:
                self.pareto_front.append(score)
                self.pareto_positions.append(self.positions[i])

def optimize_all_fault_types():
    """对每种故障类型分别进行优化"""

    
    gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
    all_results = {}
    
    # 创建输出目录
    output_base_dir = r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    for fault_type in fault_types:
        print(f"\n开始优化 {fault_type} 故障类型...")
        
        eval_file = os.path.join(
            r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\evaluation_results",
            f'evaluation_results_{fault_type}.csv'
        )
        
        if not os.path.exists(eval_file):
            print(f"警告: 找不到评估文件 {eval_file}，跳过此故障类型")
            continue
            
        try:
            # 读取评估数据，使用第一列作为索引
            eval_df = pd.read_csv(eval_file, index_col=0)
            print(f"\n当前评估数据:\n{eval_df}")
            
            def objective_functions_local(pso_variables, eval_df):
             """局部目标函数，使用当前故障类型的评估数据"""
             try:
                 gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
                 
                 # 检查所有必需的数据是否存在
                 for gan in gan_types:
                     if gan not in eval_df.index:
                         raise KeyError(f"找不到GAN类型: {gan}")
                     for metric in ['MMD', 'Coverage', 'Wasserstein', 'ModeScore']:
                         if metric not in eval_df.columns:
                             raise KeyError(f"找不到指标: {metric}")
                 
                 # 计算目标函数
                 objectives = []
                 
                 # MMD (最小化)
                 mmd_sum = sum(float(eval_df.loc[gan, 'MMD']) * w 
                              for gan, w in zip(gan_types, pso_variables))
                 objectives.append(mmd_sum)
                 
                 # Coverage (最大化)
                 coverage_sum = sum(float(eval_df.loc[gan, 'Coverage']) * w 
                                  for gan, w in zip(gan_types, pso_variables))
                 objectives.append(-coverage_sum)  # 取负是因为我们要最小化
                 
                 # Wasserstein (最小化)
                 wasserstein_sum = sum(float(eval_df.loc[gan, 'Wasserstein']) * w 
                                     for gan, w in zip(gan_types, pso_variables))
                 objectives.append(wasserstein_sum)
                 
                 # ModeScore (最大化)
                 mode_score_sum = sum(float(eval_df.loc[gan, 'ModeScore']) * w 
                                    for gan, w in zip(gan_types, pso_variables))
                 objectives.append(-mode_score_sum)  # 取负是因为我们要最小化
                 
                 return np.array(objectives)
                 
             except Exception as e:
                 print(f"计算目标函数时出错: {str(e)}")
                 print(f"当前评估数据:\n{eval_df}")
                 print(f"当前权重:", pso_variables)
                 return np.array([float('inf')] * 4)

            
            # 创建MOPSO实例，使用局部目标函数
            class LocalMOPSO(MOPSO):
                def __init__(self, eval_data, *args, **kwargs):
                    self.eval_data = eval_data
                    self.error_reported = False  # 添加错误报告标志
                    super().__init__(*args, **kwargs)
                
                def update_pareto_front(self):
                    """更新帕累托前沿"""
                    scores = []
                    valid_positions = []
                    
                    # 计算所有有效的目标函数值
                    for i, p in enumerate(self.positions):
                        try:
                            score = objective_functions_local(p, self.eval_data)
                            if not np.any(np.isinf(score)):
                                scores.append(score)
                                valid_positions.append(p)
                        except Exception as e:
                            if not self.error_reported:
                                print(f"计算目标函数时出错: {str(e)}")
                                print("这个错误只显示一次，后续的相同错误将被抑制")
                                self.error_reported = True
                            continue
                    
                    # 如果没有有效解，返回空的帕累托前沿
                    if not scores:
                        self.pareto_front = []
                        self.pareto_positions = []
                        return
                        
                    # 找到非支配解
                    self.pareto_front = []
                    self.pareto_positions = []
                    
                    for i, score in enumerate(scores):
                        dominated = False
                        for other_score in scores:
                            if self.dominates(other_score, score):
                                dominated = True
                                break
                        if not dominated:
                            self.pareto_front.append(score)
                            self.pareto_positions.append(valid_positions[i])
                    
                    # 如果找到了新的帕累托解，打印信息
                    if self.pareto_front and len(self.pareto_front) > 0:
                        print(f"\n发现 {len(self.pareto_front)} 个帕累托最优解")
            
            # 运行优化
            mopso = LocalMOPSO(
                eval_data=eval_df,
                num_particles=50,
                num_dimensions=len(gan_types),
                num_objectives=4,
                max_iter=200,
                w=0.5,
                c1=1.5,
                c2=1.5
            )
            
            pareto_positions, pareto_front = mopso.optimize()
            
            # 保存结果
            results_df = pd.DataFrame()
            
            # 添加权重
            for i, gan_type in enumerate(gan_types):
                results_df[f'Weight_{gan_type}'] = [pos[i] for pos in pareto_positions]
            
            # 添加目标函数值
            results_df['MMD'] = [front[0] for front in pareto_front]
            results_df['Coverage'] = [-front[1] for front in pareto_front]
            results_df['Wasserstein'] = [front[2] for front in pareto_front]
            results_df['ModeScore'] = [-front[3] for front in pareto_front]
            
            # 保存到文件
            output_path = os.path.join(output_base_dir, f'mopso_results_{fault_type}.csv')
            results_df.to_csv(output_path, index=False)
            
            all_results[fault_type] = results_df
            print(f"{fault_type} 故障类型的优化结果已保存到: {output_path}")
            
        except Exception as e:
            #print(f"处理 {fault_type} 故障类型时出错: {str(e)}")
            continue
    
    # 生成总结报告
    generate_optimization_summary(all_results)

def generate_optimization_summary(all_results):
    """生成优化结果的总结报告，选择归一化评价指标总和最大的解"""
    # 创建一个空的DataFrame来存储结果
    summary_df = pd.DataFrame()
    gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
    
    for fault_type, results in all_results.items():
        try:
            # 检查结果是否为空
            if results.empty:
                print(f"警告: {fault_type} 的结果为空，跳过")
                continue
                
            # 处理NaN值
            results = results.fillna(0)  # 将NaN值填充为0
            
            # 归一化四个评价指标
            normalized_scores = results.copy()
            
            # MMD和Wasserstein是越小越好，需要反向归一化
            for col in ['MMD', 'Wasserstein']:
                if col in results.columns:
                    min_val = results[col].min()
                    max_val = results[col].max()
                    if max_val - min_val != 0:
                        normalized_scores[col] = (max_val - results[col]) / (max_val - min_val)
                    else:
                        normalized_scores[col] = 1
            
            # Coverage和ModeScore是越大越好，直接归一化
            for col in ['Coverage', 'ModeScore']:
                if col in results.columns:
                    min_val = results[col].min()
                    max_val = results[col].max()
                    if max_val - min_val != 0:
                        normalized_scores[col] = (results[col] - min_val) / (max_val - min_val)
                    else:
                        normalized_scores[col] = 1
            
            # 检查所需的列是否存在
            score_columns = ['MMD', 'Coverage', 'Wasserstein', 'ModeScore']
            available_columns = [col for col in score_columns if col in normalized_scores.columns]
            
            # 计算总分
            total_scores = normalized_scores[available_columns].sum(axis=1)
            
            # 找出总分最高的解的索引
            best_index = total_scores.idxmax()
            
            # 记录最佳权重
            for gan_type in gan_types:
                weight_col = f'Weight_{gan_type}'
                if weight_col in results.columns:
                    summary_df.at[fault_type, weight_col] = results.loc[best_index, weight_col]
                else:
                    summary_df.at[fault_type, weight_col] = 0
            
            # 记录原始指标值
            for metric in score_columns:
                if metric in results.columns:
                    summary_df.at[fault_type, metric] = results.loc[best_index, metric]
                else:
                    summary_df.at[fault_type, metric] = 0
                    
            summary_df.at[fault_type, 'Total_Score'] = total_scores[best_index]
            
        except Exception as e:
            print(f"处理 {fault_type} 时出错: {str(e)}")
            continue
    
    try:
        # 保存总结报告
        summary_path = os.path.join(
            r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\mopso_results",
            'optimization_summary.csv'
        )
        summary_df.to_csv(summary_path)
        print(f"\n优化总结报告已保存到: {summary_path}")
        
        # 打印每种故障类型的最优解
        for fault_type in summary_df.index:
            print(f"\n{fault_type} 故障类型的最优解:")
            print("权重:")
            for gan_type in gan_types:
                weight = summary_df.loc[fault_type, f'Weight_{gan_type}']
                print(f"{gan_type}: {weight:.4f}")
            print("评价指标:")
            for metric in score_columns:
                print(f"{metric}: {summary_df.loc[fault_type, metric]:.4f}")
            print(f"总分: {summary_df.loc[fault_type, 'Total_Score']:.4f}")
            
    except Exception as e:
        print(f"保存和打印结果时出错: {str(e)}")

def verify_evaluation_file(fault_type):
    """验证评估文件的格式和内容"""
    results_path = os.path.join(
        r"C:\Users\yeyue\Desktop\实验室工作用\论文2\Paper_Code\gan_code\evaluation_results",
        f'evaluation_results_{fault_type}.csv'
    )
    
    try:
        # 读取CSV文件
        df = pd.read_csv(results_path)
        print(f"\n验证 {fault_type} 的评估文件:")
        
        # 1. 检查基本结构
        print("\n1. 基本结构:")
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {df.columns.tolist()}")
        
        # 2. 检查数据类型
        print("\n2. 数据类型:")
        print(df.dtypes)
        
        # 3. 检查是否有缺失值
        print("\n3. 缺失值检查:")
        print(df.isnull().sum())
        
        # 4. 检查必需的GAN类型
        required_gans = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
        first_col = df.columns[0]
        existing_gans = df[first_col].values
        print("\n4. GAN类型检查:")
        print(f"现有GAN类型: {existing_gans}")
        missing_gans = [gan for gan in required_gans if gan not in existing_gans]
        if missing_gans:
            print(f"警告: 缺少以下GAN类型: {missing_gans}")
        
        # 5. 检查数值范围
        print("\n5. 数值范围检查:")
        for col in ['MMD', 'Coverage', 'Wasserstein', 'ModeScore']:
            if col in df.columns:
                print(f"{col}: 最小值={df[col].min():.4f}, 最大值={df[col].max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"验证评估文件时出错: {str(e)}")
        return False



if __name__ == "__main__":
    # 首先验证所有评估文件
    all_valid = True
    for fault_type in fault_types:
        if not verify_evaluation_file(fault_type):
            all_valid = False
            print(f"警告: {fault_type} 的评估文件验证失败")
    
    if not all_valid:
        print("\n请修复评估文件后再继续")
    else:
        # 运行优化
        optimize_all_fault_types()

def objective_functions_local(pso_variables, eval_df):
    """局部目标函数，使用当前故障类型的评估数据"""
    try:
        gan_types = ['CTGAN', 'DCGAN', 'infoGAN', 'LSGAN', 'WGAN_GP']
        
        # 检查所有必需的数据是否存在
        for gan in gan_types:
            if gan not in eval_df.index:
                raise KeyError(f"找不到GAN类型: {gan}")
            for metric in ['MMD', 'Coverage', 'Wasserstein', 'ModeScore']:
                if metric not in eval_df.columns:
                    raise KeyError(f"找不到指标: {metric}")
        
        # 计算目标函数
        objectives = []
        
        # MMD (最小化)
        mmd_sum = sum(float(eval_df.loc[gan, 'MMD']) * w 
                     for gan, w in zip(gan_types, pso_variables))
        objectives.append(mmd_sum)
        
        # Coverage (最大化)
        coverage_sum = sum(float(eval_df.loc[gan, 'Coverage']) * w 
                         for gan, w in zip(gan_types, pso_variables))
        objectives.append(-coverage_sum)  # 取负是因为我们要最小化
        
        # Wasserstein (最小化)
        wasserstein_sum = sum(float(eval_df.loc[gan, 'Wasserstein']) * w 
                            for gan, w in zip(gan_types, pso_variables))
        objectives.append(wasserstein_sum)
        
        # ModeScore (最大化)
        mode_score_sum = sum(float(eval_df.loc[gan, 'ModeScore']) * w 
                           for gan, w in zip(gan_types, pso_variables))
        objectives.append(-mode_score_sum)  # 取负是因为我们要最小化
        
        return np.array(objectives)
        
    except Exception as e:
        print(f"计算目标函数时出错: {str(e)}")
        print(f"当前评估数据:\n{eval_df}")
        print(f"当前权重:", pso_variables)
        return np.array([float('inf')] * 4)

class LocalMOPSO(MOPSO):
    def __init__(self, eval_data, *args, **kwargs):
        self.eval_data = eval_data
        self.error_reported = False
        super().__init__(*args, **kwargs)
    
    def update_pareto_front(self):
        """更新帕累托前沿"""
        try:
            scores = []
            valid_positions = []
            
            for pos in self.positions:
                try:
                    score = objective_functions_local(pos, self.eval_data)
                    if not np.any(np.isinf(score)):
                        scores.append(score)
                        valid_positions.append(pos)
                except Exception as e:
                    if not self.error_reported:
                        print(f"计算粒子目标函数时出错: {str(e)}")
                        self.error_reported = True
                    continue
            
            if not scores:
                return
            
            # 更新帕累托前沿
            self.pareto_front = []
            self.pareto_positions = []
            
            for i, score in enumerate(scores):
                is_dominated = False
                j = 0
                while j < len(scores) and not is_dominated:
                    if i != j and self.dominates(scores[j], score):
                        is_dominated = True
                    j += 1
                
                if not is_dominated:
                    self.pareto_front.append(score)
                    self.pareto_positions.append(valid_positions[i])
            
            if len(self.pareto_front) > 0:
                print(f"\n找到 {len(self.pareto_front)} 个帕累托最优解")
                
        except Exception as e:
            print(f"更新帕累托前沿时出错: {str(e)}")
