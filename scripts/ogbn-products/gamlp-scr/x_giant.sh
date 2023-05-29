dataset=ogbn-products
model=SAGN_SCR
suffix=main_X_giant

output_dir=out/${dataset}/${model}/${suffix}
ckpt_dir=${output_dir}/ckpt

mkdir -p ${output_dir}
mkdir -p ${ckpt_dir}

giant_x_dir=../data/giant/${dataset}/X.all.xrt-emb.npy

python pre_processing.py --num_hops 5 --dataset ogbn-products --giant_path " "

python main.py
--use-rlu
--method R_GAMLP_RLU
--stages 800
--train-num-epochs 0
--input-drop 0.2
--att-drop 0.5
--label-drop 0
--pre-process
--residual
--dataset ogbn-products
--num-runs 10
--eval 10
--act leaky_relu
--batch_size 100000
--patience 300
--n-layers-1 4
--n-layers-2 4
--bns
--gama 0.1
--tem 0.5
--lam 0.5
--ema
--mean_teacher
--ema_decay 0.99 --lr 0.001 --adap --gap 10 --warm_up 150 --kl --kl_lam 0.2 --hidden 256 --down 0.7 --top 0.9 --giant
