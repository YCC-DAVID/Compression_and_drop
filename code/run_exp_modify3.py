import concurrent.futures
import os
import multiprocessing
# from nvidia import nvcomp

def run_task(param,gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running task with {param} on GPU {gpu_id}")
    os.system(f"python /scratch/cy65664/workDir/comp_and_drop/code/modify_model5.py {param}")
    print(f"Task with {param} completed on GPU {gpu_id}")
params = [
    


            # "-epo 160 -fzepo 30 -p 5 -tol 1e2 -cmp",
            # "-epo 160 -fzepo 60 -p 10 -tol 1e2 -cmp",


            # "-epo 160 -fzepo 30 -p 5 -tol 1e2 -cmp",
            # "-epo 160 -fzepo 60 -p 10 -tol 1e2 -cmp",

            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m psnr",        
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m psnr",           
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.0 -m ssim",

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
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.3 -m ssim",
            "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim",
            "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim",
            "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.0 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.1 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.1 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.1 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.1 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.1 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.1 -m ssim",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.1 -m psnr",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.1 -m psnr",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.1 -m psnr",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.1 -m ssim",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.1 -m ssim",
            # "-epo 160 -fzepo 60 -drp 10 -tol 1e2 -gma 0.1 -m ssim",

            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.5 -m psnr",
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.5 -m psnr",
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.5 -m psnr",
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.1 -m ssim",        
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.1 -m ssim",           
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.1 -m ssim",
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.5 -m ssim",
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.5 -m ssim",
            # "-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.5 -m ssim",

            
            


            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.5 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.5 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.5 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.5 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.5 -m psnr",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.3 -m ssim",
            # "-epo 160 -fzepo 30 -drp 5 -tol 1e2 -gma 0.5 -m ssim",

         
          ]
available_gpus = [0, 1, 2, 3]
with concurrent.futures.ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
    futures = [executor.submit(run_task, param, available_gpus[i % len(available_gpus)]) for i, param in enumerate(params)]

concurrent.futures.wait(futures)
print("All tasks are done.")