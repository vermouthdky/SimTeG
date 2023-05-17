lm_types=("all-MiniLM-L6-v2" "all-mpnet-base-v2" "all-roberta-large-v1" "e5-large")
for lm in ${lm_types[@]}; do
    bash scripts/hp_search/peft_lm.sh ogbn-arxiv $lm
done

for lm in $lm_types; do
    bash scripts/hp_search/gnn.sh ogbn-arxiv $lm GAMLP optuna_peft &
    bash scripts/hp_search/gnn.sh ogbn-arxiv $lm SAGN optuna_peft &
    wait
    echo "Finished GAMLP SAGN on X-$lm"
done

bash scripts/ogbn-arxiv/revgat/hp_search.sh
