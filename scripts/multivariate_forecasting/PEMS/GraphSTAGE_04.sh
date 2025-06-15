export CUDA_VISIBLE_DEVICES=0

model_name=GraphSTAGE

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 1 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 5min\
  --lradj type1 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 1 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 5min\
  --lradj type1 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 1 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 5min\
  --lradj type1 \
  --use_norm 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --patch_seg_len 3 \
  --patch_stride_len 2 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 64 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --enable_intra_GrAG 1 \
  --enable_inter_GrAG 1 \
  --plot_graphes 0 \
  --itr 1 \
  --freq 5min\
  --lradj type1 \
  --use_norm 0 \



