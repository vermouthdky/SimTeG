dataset=ogbn-arxiv
model=revgat

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_fix

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 0 \
    --dropout 0.41 \
    --edge-drop 0.27 \
    --group 1 \
    --input-drop 0.17 \
    --label_smoothing_factor 0.01 \
    --n-heads 2 \
    --n-hidden 256 \
    --n-label-iters 1 \
    --n-layers 2 \
    --use-labels \
    --suffix ${suffix} \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}_fix

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 1 \
    --dropout 0.75 \
    --edge-drop 0.38 \
    --group 2 \
    --input-drop 0.56 \
    --label_smoothing_factor 0.52 \
    --n-heads 4 \
    --n-hidden 256 \
    --n-label-iters 2 \
    --n-layers 2 \
    --use-labels \
    --suffix ${suffix} \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}_fix

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 2 \
    --dropout 0.58 \
    --edge-drop 0.46 \
    --group 1 \
    --input-drop 0.37 \
    --label_smoothing_factor 0.02 \
    --n-heads 2 \
    --n-hidden 256 \
    --n-label-iters 2 \
    --n-layers 2 \
    --use-labels \
    --suffix ${suffix} \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt
