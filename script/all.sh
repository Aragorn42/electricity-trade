export CUDA_VISIBLE_DEVICES=4

current_time=$(date +'%H:%M:%S')
echo $current_time

for model in '200M';do
    python -u main.py \
    --model_type=$model \
    --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
    --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
    --seq_len=720 \
    --pred_len=120 \
    --batchsize=64 \
    --two_variate=False \
    --eval_day=209
done

for model in '200MTime';do
    python -u main.py \
    --model_type=$model \
    --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
    --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
    --seq_len=2880 \
    --pred_len=120 \
    --batchsize=64 \
    --two_variate=False \
    --eval_day=209
done

current_time=$(date +'%H:%M:%S')
echo $current_time