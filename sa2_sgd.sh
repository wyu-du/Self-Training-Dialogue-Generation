# sgd_restaurants/sgd_hotels/sgd_flights/sgd_calendar/sgd_banks/sgd_weather/sgd_buses/sgd_events
# sgd_homes/sgd_media/sgd_movies/sgd_music/sgd_rentalcars/sgd_ridesharing/sgd_services/sgd_travel
DOMAIN=$1
FLAG=$2   # sa2
LR=$3     # learning rate: 1e-6, 2e-5
EPOCH=$4  # training epoch: 20
CUDA=$5

cd src
for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=${CUDA} python sa2_self_training.py --timestamp=${i} \
    --learning_rate=${LR} \
    --epoch=${EPOCH} \
    --flag=${FLAG} \
    --num_dropouts=10 \
    --dataset='FewShotSGD' \
    --dataset2='sgd_amr' \
    --domain=${DOMAIN} \
    --top_p=0.9 
done
