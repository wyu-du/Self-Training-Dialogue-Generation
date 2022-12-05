DATASET=$1  # FewShotWoz/FewShotSGD
DOMAIN=$2   # restaurant/laptop/hotel/tv/attraction/train/taxi
CUDA=$3

cd src
CUDA_VISIBLE_DEVICES=${CUDA} python sa2_inference.py \
    --dataset=${DATASET} \
    --domain=${DOMAIN} \
    --top_p=0.9 