dataset=ogbn-products
model=SAGN_SCR

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/optuna/best/cached_embs/x_embs.pt
python -m src.misc.scr.pre_processing \
    --lm_model_type $lm_model_type \
    --num_hops 3 \
    --dataset ogbn-products \
    --bert_x_dir $bert_x_dir \
    --output_dir $output_dir

# python -m src.misc.scr.main --method SAGN \
#     --lm_model_type $lm_model_type \
#     --output_dir $output_dir \
#     --gpu 0 \
#     --stages 400 \
#     --train-num-epochs 0 \
#     --input-drop 0.2 \
#     --att-drop 0.4 \
#     --pre-process \
#     --residual \
#     --dataset ogbn-products \
#     --num-runs 10 \
#     --eval 10 \
#     --batch_size 50000 \
#     --patience 300 \
#     --tem 0.5 \
#     --lam 0.5 \
#     --ema \
#     --mean_teacher \
#     --ema_decay 0.0 \
#     --lr 0.001 \
#     --adap \
#     --gap 20 \
#     --warm_up 100 \
#     --top 0.85 \
#     --down 0.8 \
#     --kl \
#     --kl_lam 0.2 \
#     --hidden 256 \
#     --zero-inits \
#     --dropout 0.5 \
#     --num-heads 1 \
#     --label-drop 0.5 \
#     --mlp-layer 1 \
#     --num_hops 3 \
#     --label_num_hops 9 \
#     --disable_tqdm \
#     --label_smoothing_factor 0.3 \
#     --use_bert_x \
#     2>&1 | tee ${output_dir}/log.txt &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/optuna/best/cached_embs/x_embs.pt
python -m src.misc.scr.pre_processing \
    --lm_model_type $lm_model_type \
    --num_hops 3 \
    --dataset ogbn-products \
    --bert_x_dir $bert_x_dir \
    --output_dir $output_dir

# python -m src.misc.scr.main --method SAGN \
#     --lm_model_type $lm_model_type \
#     --output_dir $output_dir \
#     --gpu 1 \
#     --stages 400 \
#     --train-num-epochs 0 \
#     --input-drop 0.2 \
#     --att-drop 0.4 \
#     --pre-process \
#     --residual \
#     --dataset ogbn-products \
#     --num-runs 10 \
#     --eval 10 \
#     --batch_size 50000 \
#     --patience 300 \
#     --tem 0.5 \
#     --lam 0.5 \
#     --ema \
#     --mean_teacher \
#     --ema_decay 0.0 \
#     --lr 0.001 \
#     --adap \
#     --gap 20 \
#     --warm_up 100 \
#     --top 0.85 \
#     --down 0.8 \
#     --kl \
#     --kl_lam 0.2 \
#     --hidden 256 \
#     --zero-inits \
#     --dropout 0.5 \
#     --num-heads 1 \
#     --label-drop 0.5 \
#     --mlp-layer 1 \
#     --num_hops 3 \
#     --label_num_hops 9 \
#     --disable_tqdm \
#     --label_smoothing_factor 0.3 \
#     --use_bert_x \
#     2>&1 | tee ${output_dir}/log.txt &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/optuna/best/cached_embs/x_embs.pt
python -m src.misc.scr.pre_processing \
    --lm_model_type $lm_model_type \
    --num_hops 3 \
    --dataset ogbn-products \
    --bert_x_dir $bert_x_dir \
    --output_dir $output_dir

# python -m src.misc.scr.main --method SAGN \
#     --lm_model_type $lm_model_type \
#     --output_dir $output_dir \
#     --gpu 2 \
#     --stages 400 \
#     --train-num-epochs 0 \
#     --input-drop 0.2 \
#     --att-drop 0.4 \
#     --pre-process \
#     --residual \
#     --dataset ogbn-products \
#     --num-runs 10 \
#     --eval 10 \
#     --batch_size 50000 \
#     --patience 300 \
#     --tem 0.5 \
#     --lam 0.5 \
#     --ema \
#     --mean_teacher \
#     --ema_decay 0.0 \
#     --lr 0.001 \
#     --adap \
#     --gap 20 \
#     --warm_up 100 \
#     --top 0.85 \
#     --down 0.8 \
#     --kl \
#     --kl_lam 0.2 \
#     --hidden 256 \
#     --zero-inits \
#     --dropout 0.5 \
#     --num-heads 1 \
#     --label-drop 0.5 \
#     --mlp-layer 1 \
#     --num_hops 3 \
#     --label_num_hops 9 \
#     --disable_tqdm \
#     --label_smoothing_factor 0.3 \
#     --use_bert_x \
#     2>&1 | tee ${output_dir}/log.txt
