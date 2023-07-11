dataset=ogbn-products
model=SAGN_SCR
suffix=main_X_giant

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

giant_x_dir=../data/giant/${dataset}/X.all.xrt-emb.npy

# python -m src.misc.scr.pre_processing \
#     --num_hops 3 \
#     --dataset ogbn-products \
#     --giant_path $giant_x_dir \
#     --output_dir $output_dir

python -m src.misc.scr.main --method SAGN \
    --output_dir $output_dir \
    --gpu 3 \
    --stages 1000 \
    --train-num-epochs 0 \
    --input-drop 0.2 \
    --att-drop 0.4 \
    --pre-process \
    --residual \
    --dataset ogbn-products \
    --num-runs 10 \
    --eval 10 \
    --batch_size 50000 \
    --patience 300 \
    --tem 0.5 \
    --lam 0.5 \
    --ema \
    --mean_teacher \
    --ema_decay 0.0 \
    --lr 0.001 \
    --adap \
    --gap 20 \
    --warm_up 100 \
    --top 0.85 \
    --down 0.8 \
    --kl \
    --kl_lam 0.2 \
    --hidden 256 \
    --zero-inits \
    --dropout 0.5 \
    --num-heads 1 \
    --label-drop 0.5 \
    --mlp-layer 1 \
    --num_hops 3 \
    --label_num_hops 9 \
    --disable_tqdm \
    --giant \
    --giant_path $giant_x_dir \
    2>&1 | tee ${output_dir}/log.txt
