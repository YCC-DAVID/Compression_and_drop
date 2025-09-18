import concurrent.futures
import os

def run_task(param, gpu_id):
    # 为当前任务设置指定的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running task with '{param}' on GPU {gpu_id}")
    
    # 设置每个任务使用4个CPU核心
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    os.system(f"python /scratch/cy65664/workDir/comp_and_drop/code/modify_model5.py {param}")
    print(f"Task with '{param}' completed on GPU {gpu_id}")

# 任务参数列表
params = [
    # "-epo 160",
    # "-epo 160",
    # "-epo 160",
    # "-epo 160 -fzepo 30 -p 5",
    # "-epo 160 -fzepo 30 -p 5",
    # "-epo 160 -fzepo 30 -p 5",
    # "-epo 160 -fzepo 60 -p 10",
    # "-epo 160 -fzepo 60 -p 10",
    # "-epo 160 -fzepo 60 -p 10",
    # "-epo 160 -fzepo 30 60 -p 5 10",
    # "-epo 160 -fzepo 30 60 -p 5 10",
    # "-epo 160 -fzepo 30 60 -p 5 10",
    # "-epo 80",
    # "-epo 80 -fzepo 10 -p 5",
    # "-epo 80 -fzepo 20 -p 10",
    # "-epo 80 -fzepo 10 20 -p 5 10",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 2",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 4",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 8",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 16",
    # # "-epo 160 -fzepo 30 -p 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 32",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 1",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 2",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 4",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 8",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 16",
    # # # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim --cmp_batch_size 32",
    # # # "-epo 50 -fzepo 20 -p 10",
    # # "-epo 160 -fzepo 60 -p 10 -tol 1e2 -gma 0.0 -m ssim",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 1",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 2",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 4",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 8",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 16",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim --cmp_batch_size 16",
    # "-epo 50 -fzepo 10 20 -p 5 10",
    # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim",
    # "-epo 160 -fzepo 30 60 -p 5 10 -tol 1e2 -gma 0.3 -m ssim --cmp_batch_size 1",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 1",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 2",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 4",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 8",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 16",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim --cmp_batch_size 32",
]

# 可用 GPU 数量
available_gpus = [0, 1, 2, 3]
print("Computation start")

# 分配 GPU 给任务
with concurrent.futures.ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
    futures = [executor.submit(run_task, param, available_gpus[i % len(available_gpus)]) for i, param in enumerate(params)]

# 等待所有任务完成
concurrent.futures.wait(futures)
print("All tasks are done.")
