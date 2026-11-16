# CUDA_VISIBLE_DEVICES=0,3,4,5 python -m  torch.distributed.run --nnodes 1 --nproc_per_node 4  train.py --config ./configs/cod-sam-vit-l_allmoe.yaml
# CUDA_VISIBLE_DEVICES=0,3,7 python -m  torch.distributed.run --nnodes 1 --nproc_per_node 3  train.py --config ./configs/cod-sam-vit-l_allmoe.yaml
# CUDA_VISIBLE_DEVICES=4,7 nohup python -m  torch.distributed.run --nnodes 1 --nproc_per_node 2  train.py --config ./configs/cod-sam-vit-all-film.yaml > film_all.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=4,7 nohup python -m  torch.distributed.run --nnodes 1 --nproc_per_node 2  train.py --config ./configs/cod-sam-vit-all-base.yaml > base_all.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=3,4 nohup python -m  torch.distributed.run --nnodes 1 --nproc_per_node 2  train.py --config ./configs/cod-sam-vit-fusion_all.yaml > fusion_all.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1,7 nohup python -m  torch.distributed.run --nnodes 1 --nproc_per_node 2 --master_port=29617 train.py --config ./configs/cod-sam-vit-fusion_all.yaml > fusion_all.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=5,6 nohup python -m  torch.distributed.run --nnodes 1 --nproc_per_node 2 --master_port=29617 train.py --config ./configs/cod-sam-vit-fusion_aigc.yaml > fusion_aigc.log 2>&1 & 

######ps#######################################

# CUDA_VISIBLE_DEVICES=4,7 nohup python -m torch.distributed.run \
#   --nnodes=1 \
#   --nproc_per_node=2 \
#   --master_port=29617 \
#   train.py \s
#   --config ./configs/cod-sam-vit-pmpt.yaml \
#   > tpmpt_ps.log 2>&1 &