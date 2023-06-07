bash scripts/hp_search/peft_lm.sh ogbn-products all-MiniLM-L6-v2
# bash scripts/hp_search/peft_lm.sh ogbn-products all-mpnet-base-v2
# bash scripts/hp_search/peft_lm.sh ogbn-products all-roberta-large-v1
# bash scripts/hp_search/peft_lm.sh ogbn-products e5-large

# for lm in all-MiniLM-L6-v2 all-roberta-large-v1 e5-large; do
#     bash scripts/hp_search/gnn.sh ogbn-products $lm GAMLP optuna &
#     bash scripts/hp_search/gnn.sh ogbn-products $lm SAGN optuna
#     echo "Finished GAMLP SAGN on X-$lm"
# done

# for lm in all-MiniLM-L6-v2 all-roberta-large-v1 e5-large; do
#     bash scripts/hp_search/gnn.sh ogbn-products $lm GAMLP optuna_peft &
#     bash scripts/hp_search/gnn.sh ogbn-products $lm SAGN optuna_peft &
#     wait
#     echo "Finished GAMLP SAGN on X-$lm"
# done
