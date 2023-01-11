# dataset and test mode
dataset=$1
ckpt_name=$2 # TGRoberta-best.pt
mode='test'

# training parameters
ckpt_dir='./ckpt'

# cont
cont=1

python main.py \
    --mode $mode \
    --dataset $dataset \
    --cont $cont \
    --ckpt_dir $ckpt_dir \
    --ckpt_name $ckpt_name 2>&1 | tee logs/ogbn-arxiv-test.txt
