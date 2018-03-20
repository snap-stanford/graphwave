#!/bin/bash

# run.sh

# -------------------------------------------------------------------------------------
# Run on synthetic Erdos-Renyi graphs

mkdir -p {_data,_results}/synthetic

# --
# Compute features for all nodes in small synthetic graph 
python utils/make-graph.py --n-nodes 3200 --outpath ./_data/synthetic/3200.edgelist

# serial -- 15 seconds
python main.py \
    --inpath ./_data/synthetic/3200.edgelist \
    --outpath ./_results/synthetic/3200

# parallel -- 2 seconds
python main.py \
    --n-chunks 16 --n-jobs 16 \
    --inpath ./_data/synthetic/3200.edgelist \
    --outpath ./_results/synthetic/3200

# --
# Compute features for sample of nodes in larger synthetic graph
python utils/make-graph.py --n-nodes 51200 --outpath ./_data/synthetic/51200.edgelist

# parallel -- 15 seconds
python main.py \
    --n-jobs 32 --n-chunks 32 \
    --inpath ./_data/synthetic/51200.edgelist \
    --outpath ./_results/synthetic/51200 \
    --n-queries 512

# -------------------------------------------------------------------------------------
# Run on POKEC graph (~ 3 minutes)

mkdir -p {_data,_results}/pokec
wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v0/generic/pokec.edgelist.gz
gunzip pokec.edgelist.gz && mv pokec.edgelist ./_data/pokec

python main.py \
    --n-jobs 16 --n-chunks 16 \
    --inpath ./_data/pokec/pokec.edgelist \
    --outpath ./_results/pokec/pokec \
    --n-queries 64 

# -------------------------------------------------------------------------------------
# Run on larger graphs

# a) could run on `pokec` graph with a larger `--num-queries` parameter
# b) could run on a larger graph