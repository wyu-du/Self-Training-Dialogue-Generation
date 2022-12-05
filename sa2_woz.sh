DOMAIN=$1 # restaurant/laptop/hotel/tv/attraction/train/taxi
FLAG=$2   # sa2
LR=$3     # learning rate: 1e-6, 1e-5
EPOCH=$4  # 20, 10
CUDA=$5

cd src
for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=${CUDA} python sa2_self_training.py --timestamp=${i} \
    --flag=${FLAG} \
    --learning_rate=${LR} \
    --epoch=${EPOCH} \
    --num_dropouts=10 \
    --dataset='FewShotWoz' \
    --dataset2='multiwoz_amr' \
    --domain=${DOMAIN} \
    --top_p=0.9 \
    --filter_out
done
