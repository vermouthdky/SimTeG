dataset=ogbl-citation2
model=SEAL
suffix=main_X_ogb

bash scripts/ogbl-citation2/seal/seal_train.sh --model_type $model --dataset $dataset --suffix $suffix \
    --single_gpu 3 \
    --num_hops 1 \
    --use_feature \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --dynamic_train \
    --dynamic_val \
    --dynamic_test \
    --train_percent 2 \
    --val_percent 1 \
    --test_percent 1 \
    --runs 5
