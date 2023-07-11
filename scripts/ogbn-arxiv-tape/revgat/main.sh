model=revgat
dataset=ogbn-arxiv

lm_model_type=e5-large
suffix=ensemble_preds

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 0 \
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
    --use_gpt_preds \
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --save_pred \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt

dataset=ogbn-arxiv

lm_model_type=e5-large
suffix=ensemble_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 0 \
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
    --save_pred \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt &

dataset=ogbn-arxiv-tape
lm_model_type=e5-large
suffix=ensemble_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 1 \
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
    --save_pred \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt &

dataset=ogbn-arxiv
lm_model_type=all-roberta-large-v1
suffix=ensemble_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 1 \
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
    --save_pred \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt &

dataset=ogbn-arxiv-tape
lm_model_type=all-roberta-large-v1
suffix=ensemble_X_${lm_model_type}

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt

python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 1 \
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
    --save_pred \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt

# logits1=out/ogbn-arxiv/revgat/ensemble_X_e5-large/cached_embs
# logits2=out/ogbn-arxiv/revgat/ensemble_X_all-roberta-large-v1/cached_embs
# logits3=out/ogbn-arxiv/revgat/ensemble_X_e5-large-v2/cached_embs

# logits4=out/ogbn-arxiv-tape/revgat/ensemble_X_e5-large/cached_embs
# logits5=out/ogbn-arxiv-tape/revgat/ensemble_X_all-roberta-large-v1/cached_embs
# logits6=out/ogbn-arxiv-tape/revgat/ensemble_X_e5-large-v2/cached_embs

# logits7=out/ogbn-arxiv/revgat/ensemble_preds/cached_embs

# python compute_ensemble.py \
#     --list_logits "${logits1} ${logits2} ${logits4} ${logits5} ${logits7}" \
#     --weights 2 2 1 1 1 \
#     --start_seed 1
