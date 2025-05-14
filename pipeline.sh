#!/bin/bash

DATASET_NAME="physics"
TRANS_MODE="transh"
MODEL_SAVE_NAME="all_minilm"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_NAME="sentence-transformers/all-MiniLM-L6-v2"

# This script downloads the datasets, unzips them
wget https://zenodo.org/records/6606557/files/${DATASET_NAME}.zip?download=1 -O ${DATASET_NAME}.zip
unzip ${DATASET_NAME}.zip
rm ${DATASET_NAME}.zip

python create_kg.py --dataset_folder ${DATASET_NAME} 

cd src

# Define variables
MAX_TOKENS=256
NORMALIZE="--normalize"
POOLING_MODE="mean"
DEVICE="cuda"
DATASET_FOLDER="../${DATASET_NAME}"
BATCH_SIZE=256
SHUFFLE="--shuffle"
MAX_EPOCH=10
LR=2e-5
MODEL_DIR="../models"
LOSS_MARGIN=0.1
EMBEDDING_SIZE=384
EMBEDDING_DIR="../embeddings"
RUN_DIR="../runs"

# Train the bi-encoder
python train_bi_encoder.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --model_dir "${MODEL_DIR}" \
    --model_name "${MODEL_NAME}" \
    --model_save_name "${MODEL_SAVE_NAME}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --batch_size ${BATCH_SIZE} \
    ${SHUFFLE} \
    --max_tokens ${MAX_TOKENS} \
    ${NORMALIZE} \
    --pooling_mode "${POOLING_MODE}" \
    --lr ${LR} \
    --epoch ${MAX_EPOCH} \
    --loss_gamma ${LOSS_MARGIN}

# Create embeddings
python create_bi_encoder_embeddings.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --embedding_dir "${EMBEDDING_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --model_name "${MODEL_NAME}" \
    --model_save_name "${MODEL_SAVE_NAME}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --embedding_size ${EMBEDDING_SIZE} \
    --max_tokens ${MAX_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    ${NORMALIZE} \
    --pooling_mode "${POOLING_MODE}"

# Create run files
python create_run.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --model_name "${MODEL_SAVE_NAME}" \
    --embeddings_folder "${EMBEDDING_DIR}" \
    --runs_path "${RUNS_DIR}"


GRAPH_BATCH_SIZE=16384
GRAPH_LR=1e-3
GRAPH_MAX_EPOCH=100
N_RELATIONS=5

# Train the graph model
python train_graph.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --model_save_name "${MODEL_SAVE_NAME}" \
    ${SHUFFLE} \
    --batch_size ${GRAPH_BATCH_SIZE} \
    --lr ${GRAPH_LR} \
    --n_relations ${N_RELATIONS} \
    --trans_mode "${TRANS_MODE}" \
    --device "${DEVICE}" \
    --max_epoch ${GRAPH_MAX_EPOCH} \
    --embeddings_folder "${EMBEDDING_DIR}" \
    --model_dir "${MODEL_DIR}"

# Test the graph model
python test_graph.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --n_relations ${N_RELATIONS} \
    --trans_mode "${TRANS_MODE}" \
    --runs_path "${RUN_DIR}" \
    --embeddings_folder "${EMBEDDING_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --model_save_name "${MODEL_SAVE_NAME}"