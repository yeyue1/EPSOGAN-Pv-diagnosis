import os
import subprocess
import sys

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # 1. 
        print("训练LSGAN")
        subprocess.run([sys.executable, os.path.join(current_dir, 'LSGAN.py')], check=True)  
        
        
        # 2. 
        print("\n训练WGAN...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'WGAN_GP.py')], check=True)

        # 3. 
        print("\n训练infoGAN...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'infoGAN.py')], check=True)

        #4.
        print("\n训练DCGAN...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'DCGAN.py')], check=True)
        
        #5. 
        print("\n训练CTGAN...")
        subprocess.run([sys.executable, os.path.join(current_dir, 'CTGAN.py')], check=True)

        print("\n训练完成，结果已保存")
    except subprocess.CalledProcessError as e:
        print(f"执行脚本时出错: {e}")
    except Exception as e:
        print(f"运行过程中出错: {e}")

if __name__ == "__main__":
    main()
