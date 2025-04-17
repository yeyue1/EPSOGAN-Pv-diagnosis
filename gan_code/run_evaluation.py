import os
import subprocess
import sys

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # 1. 首先运行评估脚本生成评估结果
        print("正在运行GAN评估...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'evaluate_gan.py')], check=True)
        
            
        print("\n评估完成，结果已保存")
        
        # 2. 然后运行PSO优化
        print("\n正在运行PSO优化...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'EPSOGAN.py')], check=True)

        # 3. 评估最优帕累托解寻找最优数据集
        print("\n正在运行最优解计算...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'generate_weighted_data_evaluate.py')], check=True)

        #4. 生成数据集
        print("\n 生成最优解集..")
        subprocess.run([sys.executable, os.path.join(current_dir, 'generate_weighted_data.py')], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"执行脚本时出错: {e}")
    except Exception as e:
        print(f"运行过程中出错: {e}")

if __name__ == "__main__":
    main()
