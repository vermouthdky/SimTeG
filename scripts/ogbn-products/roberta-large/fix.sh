dataset=ogbn-products
model_type=all-roberta-large-v1
suffix=fix

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --eval_batch_size 200 \
    --mode "test"

bert_x_dir=out/${dataset}/${lm_type}/${suffix}/cached_embs/iter_0_x_embs.pt
