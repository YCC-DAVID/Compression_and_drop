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
CUDA_VISIBLE_DEVICES=0 python /mnt/Compression_and_drop/code/modify_model6.py -epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.3 -m ssim --cmp_batch_size 8
```
