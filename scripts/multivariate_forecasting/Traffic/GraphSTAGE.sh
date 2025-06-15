export CUDA_VISIBLE_DEVICES=0

model_name=GraphSTAGE

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --patch_seg_len 6 \
  --patch_stride_len 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 1h \
  --lradj type1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --patch_seg_len 6 \
  --patch_stride_len 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 1h \
  --lradj type1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --patch_seg_len 6 \
  --patch_stride_len 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 1h \
  --lradj type1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --patch_seg_len 6 \
  --patch_stride_len 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 1h \
  --lradj type1 \



