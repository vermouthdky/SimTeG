dataset='ogbn-arxiv'
model_type='GAMLP'
suffix='bert_x_JK_GAMLP'

# training parameters
# searched by optuna, valid acc: 70.30 %
# with the same HPs, test on bert_x, valid_acc ~76%

#! --use_bert_x

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_interval 5 \
    --lr 0.01 \
    --weight_decay 1e-7 \
    --batch_size 10000 \
    --eval_batch_size 10000 \
    --epochs 500 \
    --accum_interval 1 \
    --gnn_dropout 0.15 \
    --label_smoothing 0.5 \
    --warmup_ratio 0.3 \
    --use_bert_x
