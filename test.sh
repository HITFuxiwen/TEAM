export PYTHONNOUSERSITE=1
# python test.py \
#   --config ./configs/cod-sam-vit-fusion_all.yaml \
#   --model ./save/_cod-sam-vit-fusion_all_globalstd/model_epoch_10.pth \
#   --save_path ./xuantu/casia

# python test.py \
#   --config ./configs/cod-sam-vit-fusion_all.yaml \
#   --model ./save/_cod-sam-vit-fusion_all_noexpert/model_epoch_9.pth \
#   --save_path ./xuantu_noexpert/imd

# python test.py \
#   --config ./configs/cod-sam-vit-fusion_all.yaml \
#   --model ./save/_cod-sam-vit-fusion_all_globalnewlr/model_epoch_17.pth \
#   --save_path ./xuantu_globalnewlr/aigc
  
python test.py \
  --config ./configs/cod-sam-vit-fusion_all.yaml \
  --model ./save/_cod-sam-vit-fusion_all/model_epoch_3.pth \
  --save_path ./xuantu_stdsamelr/dso2

# python test.py \
#   --config ./configs/cod-sam-vit-all-base.yaml \
#   --model ./save/_cod-sam-vit-all-base/model_epoch_5.pth \
#   --save_path ./results/fusion/all

# python test.py \
#   --config ./configs/cod-sam-vit-all-film.yaml \
#   --model ./save/_cod-sam-vit-all-film/model_epoch_20.pth \
#   --save_path ./results/film/aigc

# python test.py \
#   --config ./configs/cod-sam-vit-all-base.yaml \
#   --model ./save/_cod-sam-vit-all-base/model_epoch_14.pth \
#   --save_path ./results/base/aigc