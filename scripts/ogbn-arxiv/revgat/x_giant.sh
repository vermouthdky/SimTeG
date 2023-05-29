dataset=ogbn-arxiv
model=revgat

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_giant

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

giant_x_dir=../data/giant/${dataset}/X.all.xrt-emb.npy
python -m src.misc.revgat.main \
    --use-norm \
    --no-attn-dst \
    --mode teacher \
    --gpu 2 \
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
    --ckpt_dir $ckpt_dir \
    --output_dir $output_dir \
    --use_giant_x \
    --giant_x_dir $gaint_x_dir \
    --n-runs 10 \
    2>&1 | tee ${output_dir}/log.txt
