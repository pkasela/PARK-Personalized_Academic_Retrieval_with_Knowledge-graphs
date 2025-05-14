import json

import os


from indxr import Indxr
from torch import no_grad, save, zeros
from tqdm import tqdm

from model.utils import seed_everything
from model.bi_encoder_model import BiEncoder
import click

@click.command()
@click.option('--dataset_folder', default='physics', help='Dataset folder')
@click.option('--dataset_name', default='physics', help='Dataset name')
@click.option('--device', default='cpu', help='Device')
@click.option('--embedding_dir', default='../models', help='Embedding Directory')
@click.option('--model_dir', default='../models', help='Model Directory')
@click.option('--model_name', default='all_minilm', help='Model name')
@click.option('--model_save_name', default='all_minilm', help='Model save name')
@click.option('--tokenizer_name', default='all_minilm', help='tokenizer name')
@click.option('--embedding_size', default=384, type=int, help='Embedding size')
@click.option('--seed', default=42, type=int, help='Seed')
@click.option('--max_tokens', default=512, type=int, help='Max tokens')
@click.option('--batch_size', default=32, type=int, help='Batch size')
@click.option('--normalize/--no_normalize', default=True, help='Normalize')
@click.option('--pooling_mode', default='cls', help='Pooling mode')
def main(dataset_folder, dataset_name, device, embedding_dir, model_dir, model_name, model_save_name, tokenizer_name, embedding_size, seed, max_tokens, batch_size, normalize, pooling_mode) -> None:

    os.makedirs(embedding_dir, exist_ok=True)

    os.makedirs(model_dir, exist_ok=True)

    
    seed_everything(seed)

    corpus = Indxr(os.path.join(dataset_folder, 'collection.jsonl'), key_id='id')
    corpus = sorted(corpus, key=lambda k: len(k.get("title", "") + k.get("text", "")), reverse=True)
    
    model = BiEncoder(
        model_name,
        tokenizer_name,
        max_tokens,
        normalize,
        pooling_mode,
        device
    )
    model.load(os.path.join(model_dir, dataset_name,f'{model_save_name}.pt'))
    model.eval()

    index = 0
    texts = []
    id_to_index = {}
    embedding_matrix = zeros(len(corpus), embedding_size).float()

    for doc in tqdm(corpus):
        id_to_index[doc['id']] = index
        index += 1
        texts.append(doc.get('title','')) # + '. ' + doc['text'])
        if len(texts) == batch_size:
            with no_grad():
                embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()

    save(embedding_matrix, os.path.join(embedding_dir, dataset_name, f'{model_name}.pt'))

    with open(os.path.join(embedding_dir, dataset_name, f'{model_name}.json'), 'w') as f:
        json.dump(id_to_index, f)

    # query embeddings
    
    val_query_data = Indxr(os.path.join(dataset_folder, 'val' , 'queries.jsonl'), key_id='id')
    test_query_data = Indxr(os.path.join(dataset_folder, 'test' , 'queries.jsonl'), key_id='id')

    index = 0
    texts = []
    q_id_to_index = {}
    q_embedding_matrix = zeros(len(val_query_data) + len(test_query_data), embedding_size).float()
    for q in tqdm(val_query_data):
        q_id_to_index[q['id']] = index
        index += 1
        texts.append(q.get('text',''))
        if len(texts) == batch_size:
            with no_grad():
                q_embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            q_embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []

    for q in tqdm(test_query_data):
        q_id_to_index[q['id']] = index
        index += 1
        texts.append(q.get('text',''))
        if len(texts) == batch_size:
            with no_grad():
                q_embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            q_embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []
   

    save(q_embedding_matrix, os.path.join(embedding_dir, dataset_name, f'{model_name}_query.pt'))
    with open(os.path.join(embedding_dir, dataset_name, f'{model_name}_query.json'), 'w') as f:
        json.dump(q_id_to_index, f)

if __name__ == '__main__':
    main()