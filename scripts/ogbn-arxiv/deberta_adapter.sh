dataset='ogbn-arxiv'
model_type='Deberta'
suffix='adapter'
# Trial 22 finished with value: 0.7504278421401978 and parameters: {'epochs': 6, 'lr': 0.0008574702817678569, 'weight_decay': 9.928403543009874e-05, 'label_smoothing': 0.32891273528334686, 'hidden_dropout_prob': 0.14806993138716243, 'accum_interval': 1, 'adapter_hidden_size': 64, 'header_dropout_prob': 0.12381593483664642, 'warmup_ratio': 0.15934473603587057}. Best is trial 22 with value: 0.7504278421401978.

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --eval_patience 50000 \
    --lr 5e-4 \
    --weight_decay 5e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --epochs 10 \
    --accum_interval 10 \
    --hidden_dropout_prob 0.16 \
    --header_dropout_prob 0.65 \
    --label_smoothing 0.28 \
    --warmup_ratio 0.3 \
    --use_adapter \
    --use_hug_trainer
