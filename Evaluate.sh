#!/bin/bash

# Evaluate DAM in STS
# Author Mikel Artetxe (mikel.artetxe@ehu.eus)

# Set this parameters as required
STS_RUNS=1
TMP=stsbenchmark
DAM=.
vocab_size=50000

for ((seed=0;seed<$STS_RUNS;seed++)); do
    mkdir -p "$TMP/dam"
    python2 "$DAM/preprocess_datasets/preprocess-STSBenchmark.py" \
        --srcfile "$TMP/src-train.txt" --targetfile "$TMP/targ-train.txt" --labelfile "$TMP/label-train.txt" \
        --srcvalfile "$TMP/src-dev.txt" --targetvalfile "$TMP/targ-dev.txt" --labelvalfile "$TMP/label-dev.txt" \
        --srctestfile "$TMP/src-test.txt" --targettestfile "$TMP/targ-test.txt" --labeltestfile "$TMP/label-test.txt" \
        --outputfile "$TMP/dam/data" --vocabsize $vocab_size --glove "$TMP/embeddings.glove.txt" --batchsize 8 --seed $seed #>& /dev/null
    python2 "$DAM/preprocess_datasets/get_pretrain_vecs.py" \
       --glove "$TMP/embeddings.glove.txt" --outputfile "$TMP/embeddings.hdf5" \
       --dictionary "$TMP/dam/data.word.dict" --seed $seed #>& /dev/null
    python3 "$DAM/DAM/DAM_STSBenchmark_TS.py" \
        --train_file "$TMP/dam/data-train.hdf5" \
        --dev_file "$TMP/dam/data-val.hdf5" \
        --test_file "$TMP/dam/data-test.hdf5" \
        --w2v_file "$TMP/embeddings.hdf5" \
        --log_dir "$TMP/dam/" \
        --log_fname test.log\
        --gpu_id 0 \
        --epoch 25 \
        --dev_interval 1 \
        --optimizer Adam \
        --lr 0.00005 \
        --hidden_size 2000 \
        --max_length -1 \
        --display_interval 500 \
        --weight_decay 0 \
        --dropout 0 \
        --bigrams \
        --model_path "$TMP/dam/" \
        --seed $seed #\
 #       2>& 1 | tail -1 | sed 's/[^0-9.]/ /g' | sed -E 's/\s+/\t/g' | cut -f2 | tr '\n' '\t'
  #  rm -r "$TMP/dam"
done 
