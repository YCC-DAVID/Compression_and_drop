# ZO_quantization

## jetson-containers Run Command (single use)
```
jetson-containers run --volume $PWD:/mnt/Compression_and_drop --volume /run/jtop.sock:/run/jtop.sock $(autotag l4t-pytorch) 
```

## jetson-containers Run Command (persistent)
```
jetson-containers run --name cmp \
  --restart unless-stopped -d \
  --volume /home/edward/data/Compression_and_drop:/mnt/Compression_and_drop \
  --volume /run/jtop.sock:/run/jtop.sock \
  $(autotag l4t-pytorch) sleep infinity
```

## Docker Run Command (persistent)
```
docker run --runtime nvidia \
  --name cmp \
  --restart unless-stopped -d \
  --network=host \
  --volume /home/edward/data/Compression_and_drop:/mnt/Compression_and_drop \
  --volume /run/jtop.sock:/run/jtop.sock \
  dustynv/l4t-pytorch:r36.2.0 sleep infinity
```

## Attach to Docker Container
```
docker exec -it cmp bash
```

## Environment Setup
```
apt update && apt install python-is-python3 tmux -y
cd /mnt/Compression_and_drop
pip install -r requirements.txt --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

## Project Run Command
```
test 1
CUDA_VISIBLE_DEVICES=0 python /mnt/Compression_and_drop/code/modify_model6.py -epo 50 -fzepo 20 40 -p 5 10 > ./test1.log

test 2
CUDA_VISIBLE_DEVICES=0 python /mnt/Compression_and_drop/code/modify_model6.py -epo 50 -fzepo 20 40 -drp 5 10 -gma 0.2 -m ssim --cmp_batch_size 8 > ./test2.log

test 3
CUDA_VISIBLE_DEVICES=0 python /mnt/Compression_and_drop/code/modify_model6.py -epo 50 -fzepo 20 40 -drp 5 10 -gma 0.2 -m ssim --cmp_batch_size 16 > ./test3.log

test 4
CUDA_VISIBLE_DEVICES=0 python /mnt/Compression_and_drop/code/modify_model6.py -epo 50 -fzepo 20 40 -drp 5 10 -gma 0.2 -m ssim --cmp_batch_size 4 > ./test4.log

test 5
CUDA_VISIBLE_DEVICES=0 python /mnt/Compression_and_drop/code/modify_model6.py -epo 50 -fzepo 20 40 -drp 5 10 -gma 0.2 -m ssim --cmp_batch_size 8 > ./test5.log
```
