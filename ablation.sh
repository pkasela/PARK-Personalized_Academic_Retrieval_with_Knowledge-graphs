#!/bin/bash

DATASET_NAME="physics"
DATASET_FOLDER="../physics"
MODEL_SAVE_NAME="all_minilm"
SHUFFLE="--shuffle"
GRAPH_BATCH_SIZE=16384
GRAPH_LR=1e-3
TRANS_MODE="transh"
DEVICE="cuda"
GRAPH_MAX_EPOCH=100
EMBEDDING_DIR="../embeddings"
MODEL_DIR="../models"


N_RELATIONS=3
# Train the graph model with user only
python ablation_train_only_user.py \
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

# Test the graph model with user only
python ablation_test_only_user.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --n_relations ${N_RELATIONS} \
    --trans_mode "${TRANS_MODE}" \
    --runs_path "${RUN_DIR}" \
    --embeddings_folder "${EMBEDDING_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --model_save_name "${MODEL_SAVE_NAME}"


N_RELATIONS=4
# Train the graph model with user + venue
python ablation_train_w_venue.py \
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

# Test the graph model with user + venue 
python ablation_test_w_venue.py \
    --dataset_folder "${DATASET_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --n_relations ${N_RELATIONS} \
    --trans_mode "${TRANS_MODE}" \
    --runs_path "${RUN_DIR}" \
    --embeddings_folder "${EMBEDDING_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --model_save_name "${MODEL_SAVE_NAME}"
