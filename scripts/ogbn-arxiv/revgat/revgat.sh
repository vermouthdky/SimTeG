dataset=ogbn-arxiv
model=revgat
lm_model_type=all-roberta-large-v1
suffix=X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/optuna/best/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --edge-drop=0.3 \
    --input-drop=0.25 \
    --n-heads=3 \
    --n-layers 2 \
    --use-labels \
    --n-label-iters=1 \
    --dropout 0.75 \
    --n-hidden 256 \
    --n-epochs=200 \
    --save kd \
    --backbone rev \
    --group 2 \
    --mode teacher \
    --label_smoothing_factor 0.7 \
    --gpu 0 \
    --suffix ${suffix} \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    2>&1 | tee ${output_dir}/log.txt
