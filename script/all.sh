export CUDA_VISIBLE_DEVICES=1

current_time=$(date +'%H:%M:%S')
# best for chronos-2time : input_len 120*24=2880, quant 12 single, 65.09%
# chronos: input len 1440 quant 12 single, 65.03%
echo $current_time
for j in 1440;do
    for i in 12;do
        python -u main.py \
        --model_type "Chronos-2" \
        --seq_len $j \
        --pred_len 120 \
        --two_variate=True \
        --quant $i \
        --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
    done
done
# for model in 'DLinear' 'PatchTST';do
#     python -u main.py \
#     --model_type=$model \
#     --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#     --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#     --seq_len=336 \
#     --pred_len=120 \
#     --batchsize=64 \
#     --two_variate=True \
#     --eval_day=209
# done

# for model in '200MTime';do
#     python -u main.py \
#     --model_type=$model \
#     --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#     --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#     --seq_len=2880 \
#     --pred_len=120 \
#     --batchsize=64 \
#     --two_variate=False \
#     --eval_day=209
# done

current_time=$(date +'%H:%M:%S')
echo $current_time