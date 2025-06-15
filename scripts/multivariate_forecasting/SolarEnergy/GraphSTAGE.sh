export CUDA_VISIBLE_DEVICES=0

model_name=GraphSTAGE

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --enable_intra_GrAG 0 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 10min\
  --lradj type1 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --enable_intra_GrAG 0 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 10min\
  --lradj type1 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --enable_intra_GrAG 0 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 10min\
  --lradj type1 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --enable_intra_GrAG 0 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 10min\
  --lradj type1 \
  --use_norm 0 \


