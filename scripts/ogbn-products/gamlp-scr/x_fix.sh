dataset=ogbn-products
model=GAMLP_SCR

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_fix

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
python -m src.misc.scr.main --method R_GAMLP_RLU \
    --att-drop 0.5 \
    --use-rlu \
    --stages 1000 \
    --train-num-epochs 0 \
    --input-drop 0.2 \
    --label-drop 0 \
    --pre-process \
    --residual \
    --dataset ogbn-products \
    --num-runs 10 \
    --eval 10 \
    --act leaky_relu \
    --batch_size 100000 \
    --patience 300 \
    --n-layers-1 4 \
    --n-layers-2 4 \
    --bns \
    --gama 0.1 \
    --tem 0.5 \
    --lam 0.5 \
    --ema \
    --mean_teacher \
    --ema_decay 0.99 \
    --lr 0.001 \
    --adap \
    --gap 10 \
    --warm_up 150 \
    --kl \
    --kl_lam 0.2 \
    --hidden 256 \
    --down 0.7 \
    --top 0.9 \
    --disable_tqdm \
    --output_dir $output_dir \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    2>&1 | tee ${output_dir}/log.txt &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}_fix

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
python -m src.misc.scr.main --method R_GAMLP_RLU \
    --att-drop 0.5 \
    --use-rlu \
    --stages 1000 \
    --train-num-epochs 0 \
    --input-drop 0.2 \
    --label-drop 0 \
    --pre-process \
    --residual \
    --dataset ogbn-products \
    --num-runs 10 \
    --eval 10 \
    --act leaky_relu \
    --batch_size 100000 \
    --patience 300 \
    --n-layers-1 4 \
    --n-layers-2 4 \
    --bns \
    --gama 0.1 \
    --tem 0.5 \
    --lam 0.5 \
    --ema \
    --mean_teacher \
    --ema_decay 0.99 \
    --lr 0.001 \
    --adap \
    --gap 10 \
    --warm_up 150 \
    --kl \
    --kl_lam 0.2 \
    --hidden 256 \
    --down 0.7 \
    --top 0.9 \
    --disable_tqdm \
    --output_dir $output_dir \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    2>&1 | tee ${output_dir}/log.txt &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}_fix

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
python -m src.misc.scr.main --method R_GAMLP_RLU \
    --att-drop 0.5 \
    --use-rlu \
    --stages 1000 \
    --train-num-epochs 0 \
    --input-drop 0.2 \
    --label-drop 0 \
    --pre-process \
    --residual \
    --dataset ogbn-products \
    --num-runs 10 \
    --eval 10 \
    --act leaky_relu \
    --batch_size 100000 \
    --patience 300 \
    --n-layers-1 4 \
    --n-layers-2 4 \
    --bns \
    --gama 0.1 \
    --tem 0.5 \
    --lam 0.5 \
    --ema \
    --mean_teacher \
    --ema_decay 0.99 \
    --lr 0.001 \
    --adap \
    --gap 10 \
    --warm_up 150 \
    --kl \
    --kl_lam 0.2 \
    --hidden 256 \
    --down 0.7 \
    --top 0.9 \
    --disable_tqdm \
    --output_dir $output_dir \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    2>&1 | tee ${output_dir}/log.txt
