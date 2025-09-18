import os
import threading
from concurrent.futures import ProcessPoolExecutor, wait
from queue import Queue

def run_task(param, gpu_id):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        
        print(f"Running task with '{param}' on GPU {gpu_id}")
        os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} python /scratch/cy65664/workDir/comp_and_drop/code/modify_model5.py {param}")
        print(f"Task with '{param}' completed on GPU {gpu_id}")
    except Exception as e:
        print(f"Task with '{param}' failed on GPU {gpu_id}: {e}")

def task_scheduler(params, available_gpus, max_tasks_per_gpu):
    gpu_task_count = {gpu: 0 for gpu in available_gpus}
    task_queue = Queue()
    for param in params:
        task_queue.put(param)
    
    def schedule_task():
        while not task_queue.empty():
            for gpu_id in available_gpus:
                if gpu_task_count[gpu_id] < max_tasks_per_gpu:
                    param = task_queue.get()
                    gpu_task_count[gpu_id] += 1
                    futures.append(executor.submit(execute_task, param, gpu_id))
                    break
    
    def execute_task(param, gpu_id):
        try:
            run_task(param, gpu_id)
        finally:
            gpu_task_count[gpu_id] -= 1
            schedule_task()

    futures = []
    with ProcessPoolExecutor(max_workers=len(available_gpus) * max_tasks_per_gpu) as executor:
        schedule_task()
        for future in futures:
            future.result()

# 任务参数列表
params = [
    "-epo 160",
    "-epo 160",
    "-epo 160",
    "-epo 160 -fzepo 30 -p 5",
    "-epo 160 -fzepo 30 -p 5",
    "-epo 160 -fzepo 30 -p 5",
    "-epo 160 -fzepo 60 -p 10",
    "-epo 160 -fzepo 60 -p 10",
    "-epo 160 -fzepo 60 -p 10",
    "-epo 160 -fzepo 30 60 -p 5 10",
    "-epo 160 -fzepo 30 60 -p 5 10",
    "-epo 160 -fzepo 30 60 -p 5 10",
    "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.3 -m ssim",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.3 -m ssim",
]

# 可用 GPU 数量
available_gpus = [0, 1, 2, 3]

max_tasks_per_gpu = 2  # 每张 GPU 最大任务数

# 启动任务调度
task_scheduler(params, available_gpus, max_tasks_per_gpu)
print("All tasks are done.")