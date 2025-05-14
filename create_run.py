import json
import os

import torch
from tqdm import tqdm
from indxr import Indxr

from ranx import Run
import click


def get_bert_rerank(data, q_embeddings, q_id_to_index, doc_embedding, id_to_index):
    bert_run = {}
    
    for d in tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = q_embeddings[int(q_id_to_index[d['id']])].view(1,-1)

        bm25_docs = d['bm25_doc_ids']
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}

    return bert_run


dataset_folder = 'physics'
dataset_name = 'physics'
device = 'cpu'
model = 'all_minilm'
run_folder = 'runs'

@click.command()
@click.option('--dataset-folder', default='physics', help='Dataset folder')
@click.option('--dataset-name', default='physics', help='Dataset name')
@click.option('--device', default='cpu', help='Device')
@click.option('--model_name', default='all_minilm', help='Model name')
@click.option('--embeddings_folder', default='embeddings', help='Embeddings folder')
@click.option('--runs_path', default='runs', help='Runs path')
def main(dataset_folder, dataset_name, device, model_name, embeddings_folder, runs_path):
    os.makedirs(os.path.join(runs_path, dataset_name, 'val'), exist_ok=True)
    os.makedirs(os.path.join(runs_path, dataset_name, 'test'), exist_ok=True)

    doc_embedding = torch.load(os.path.join(embeddings_folder, dataset_folder, f'{model_name}.pt')).to(device)
    with open(os.path.join(embeddings_folder, dataset_folder, f'{model_name}.json'), 'r') as f:
        id_to_index = json.load(f)


    q_embeddings = torch.load(os.path.join(embeddings_folder, dataset_folder, f'{model_name}_query.pt')).to(device)
    with open(os.path.join(embeddings_folder, dataset_folder, f'{model_name}_query.json'), 'r') as f:
        q_id_to_index = json.load(f)

    split = 'val'
    query_data = Indxr(os.path.join(dataset_folder, split, 'queries.jsonl'), key_id='id')
    bert_run = get_bert_rerank(query_data, q_embeddings, q_id_to_index, doc_embedding, id_to_index)

    ranx_run = Run(bert_run, 'Neural')
    ranx_run.save(os.path.join(runs_path, dataset_name, split ,f'{model_name}.lz4'))


    split = 'test'
    query_data = Indxr(os.path.join(dataset_folder, split, 'queries.jsonl'), key_id='id')
    bert_run = get_bert_rerank(query_data, q_embeddings, q_id_to_index, doc_embedding, id_to_index)

    ranx_run = Run(bert_run, 'Neural')
    ranx_run.save(os.path.join(runs_path, dataset_name, split ,f'{model_name}.lz4'))
