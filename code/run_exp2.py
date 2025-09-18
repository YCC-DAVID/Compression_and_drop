import os
import time
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from queue import Queue

def run_task(param, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    print(f"[GPU {gpu_id}] Running task: {param}")
    process = subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES={gpu_id} python /home/chence/workspace/Cmp4ai/comp_and_drop/code/modify_model6.py {param}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if stdout:
        print(f"[GPU {gpu_id}] STDOUT:\n{stdout.decode()}")
    if stderr:
        print(f"[GPU {gpu_id}] STDERR:\n{stderr.decode()}")

    print(f"[GPU {gpu_id}] Task done: {param}")
    return gpu_id  # return GPU id to mark this one as freed

def task_scheduler(params, available_gpus, max_tasks_per_gpu):
    task_queue = Queue()
    for p in params:
        task_queue.put(p)

    # GPU 使用计数器
    gpu_task_count = {gpu: 0 for gpu in available_gpus}
    futures_map = {}  # future -> (param, gpu_id)

    with ProcessPoolExecutor(max_workers=len(available_gpus) * max_tasks_per_gpu) as executor:
        while not task_queue.empty() or futures_map:
            # 启动新的任务（如果有空闲 GPU）
            for gpu_id in available_gpus:
                if gpu_task_count[gpu_id] < max_tasks_per_gpu and not task_queue.empty():
                    param = task_queue.get()
                    future = executor.submit(run_task, param, gpu_id)
                    futures_map[future] = (param, gpu_id)
                    gpu_task_count[gpu_id] += 1

            # 检查已完成的任务
            done_futures = [f for f in futures_map if f.done()]
            for f in done_futures:
                _, gpu_id = futures_map.pop(f)
                gpu_task_count[gpu_id] -= 1

            time.sleep(1)  # 防止过于频繁地查询状态

    print("All tasks are done.")

# 启动任务
params = [
    # "-epo 160",
    # "-epo 160",
    # "-epo 160",
    # "-epo 160 -fzepo 30 -p 5",
    # "-epo 160 -fzepo 30 -p 5",
    # "-epo 160 -fzepo 30 -p 5",
    "-epo 160 -fzepo 60 -p 10",
    # "-epo 160 -fzepo 60 -p 10",
    # "-epo 160 -fzepo 60 -p 10",
    "-epo 160 -fzepo 30 60 -p 5 10",
    # "-epo 160 -fzepo 30 60 -p 5 10",
    # "-epo 160 -fzepo 30 60 -p 5 10",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
    # "-epo 160 -fzepo 30 -drp 5 -tol 1e1 -gma 0.3 -m ssim --cuda_cmp",
    "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 8",
    "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 8",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e1 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 2",
    # "-epo 160 -fzepo 60 -drp 10 -tol 1e0 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 1",
    "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 8",
    # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e1 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 2",
    # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e0 -gma 0.3 -m ssim --cuda_cmp --cmp_batch_size 1",
]

available_gpus = [2,3]
max_tasks_per_gpu = 1

task_scheduler(params, available_gpus, max_tasks_per_gpu)
print("All tasks are done.")
