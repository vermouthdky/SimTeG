dataset=ogbn-arxiv
model_type=SAGN
lm_type=all-MiniLM-L6-v2
suffix=${lm_type}/main
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 1e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.25 \
    --gnn_num_layers 6 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.22 \
    --gnn_warmup_ratio 0.36 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

suffix=${lm_type}/fix
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 1e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.25 \
    --gnn_num_layers 6 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.22 \
    --gnn_warmup_ratio 0.36 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

lm_type=all-mpnet-base-v2
suffix=${lm_type}/main
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 2e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.36 \
    --gnn_num_layers 6 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.63 \
    --gnn_warmup_ratio 0.11 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

suffix=${lm_type}/fix
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 2e-4 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.36 \
    --gnn_num_layers 6 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.63 \
    --gnn_warmup_ratio 0.11 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

lm_type=all-roberta-large-v1
suffix=${lm_type}/main
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 3e-5 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 3e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.4 \
    --gnn_num_layers 8 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.7 \
    --gnn_warmup_ratio 0.44 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10

suffix=${lm_type}/fix
bert_x_dir=out/${dataset}/${suffix}/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 3e-5 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 3e-5 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.4 \
    --gnn_num_layers 8 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.7 \
    --gnn_warmup_ratio 0.44 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir \
    --n_exps 10
