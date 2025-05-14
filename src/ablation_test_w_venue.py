import os
import json
import torch

from torch.nn import functional as F
from model.ablation_model import UserVenueGraphTransX
from tqdm import tqdm
from indxr import Indxr
from ranx import Run, Qrels, compare, fuse, optimize_fusion
import click


def get_user_rerank_zero_author(data, model, doc_id_to_user, user_id_to_index, device, top_k=1000):
    bert_run = {}
    model.eval()
    for query in tqdm(data, total=len(data)):
        q_user_id = user_id_to_index[query['user_id']]
        with torch.no_grad():
            q_embedding = F.normalize(model.author_embedding(torch.tensor([q_user_id]).to(device)), -1)

        bm25_docs = query['bm25_doc_ids']
        """
        d_embeddings = []
        for docs in bm25_docs[:top_k]:
            try:
                with torch.no_grad():
                    d_embeddings.append(model.author_embedding(torch.tensor([user_id_to_index[doc_id_to_user[docs][0]]]).to(device)))
            except KeyError:
                d_embeddings.append(torch.zeros(model.embedding_size).view(1,-1).to(device))
        d_embeddings = torch.vstack(d_embeddings)
        """
        with torch.no_grad():
            d_embeddings = model.author_embedding( torch.tensor([user_id_to_index[doc_id_to_user[x][0]] for x in bm25_docs[:top_k]]).to(device) )
            d_embeddings = F.normalize(d_embeddings, -1)
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[query['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs[:top_k])}                

    return bert_run


def get_bert_rerank_self_citation(data, model, doc_id_to_user, user_id_to_index, top_k=1000):
    bert_run = {}
    for query in tqdm(data, total=len(data)):
        q_user_id = query['user_id']
        bm25_docs = query['bm25_doc_ids']
        bert_scores = [1 if q_user_id == doc_id_to_user[x][0]  else 0 for x in bm25_docs[:top_k]]
        bert_run[query['id']] = {doc_id: bert_scores[i] for i, doc_id in enumerate(bm25_docs[:top_k])}                

    return bert_run


def get_bert_rerank(data, model, doc_id_to_user, user_id_to_index, device, top_k=1000):
    bert_run = {}
    model.eval()
    for query in tqdm(data, total=len(data)):
        q_user_id = user_id_to_index[query['user_id']]
        with torch.no_grad():
            q_embedding = model.author_embedding(torch.tensor([q_user_id]).to(device))
            q_embedding = F.normalize(q_embedding, dim=-1)

        bm25_docs = query['bm25_doc_ids']
        bert_run[query['id']] = {}
        d_embeddings = []
        for i, doc_id in enumerate(bm25_docs[:top_k]):
            with torch.no_grad():
                d_embedding = model.author_embedding( torch.tensor([user_id_to_index[u] for u in doc_id_to_user[doc_id]]).to(device) )
            d_embedding = d_embedding.mean(dim=0, keepdim=True)
            d_embeddings.append(d_embedding)
            # bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
            # bert_run[query['id']][doc_id] = bert_scores[0].cpu().detach()
        
        d_embeddings = torch.vstack(d_embeddings)
        d_embeddings = F.normalize(d_embeddings, dim=-1)
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[query['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs[:top_k])}                
        
    return bert_run



@click.command()
@click.option('--dataset_folder', default='political_science', help='Dataset folder')
@click.option('--dataset_name', default='political_science', help='Dataset name')
@click.option('--device', default='cpu', help='Device')
@click.option('--n_relations', default=5, type=int, help='Number of relations')
@click.option('--trans_mode', default='transh', help='Trans mode')
@click.option('--runs_path', default='runs', help='Runs path')
@click.option('--embeddings_folder', default='../embeddings', help='Embeddings folder')
@click.option('--model_dir', default='../models', help='Model name')
@click.option('--model_save_name', default='all_minilm', help='Model name')
def main(dataset_folder, dataset_name, device, n_relations, trans_mode, runs_path, embeddings_folder, model_dir, model_save_name):
    doc_embs = torch.load(os.path.join(embeddings_folder, dataset_name, f'{model_save_name}.pt')).to(device)
    with open(os.path.join(dataset_folder, 'user_id_to_index_to_index.json'), 'r') as f:
        user_id_to_index = json.load(f)

    with open(os.path.join(dataset_folder, 'venue_id_to_index.json'), 'r') as f:
        venue_id_to_index = json.load(f)

    with open(os.path.join('embeddings', dataset_name, 'all_minilm.json'), 'r') as f:
        doc_id_to_index = json.load(f)


    model = UserVenueGraphTransX(
        n_authors=len(user_id_to_index),
        n_venues=len(venue_id_to_index),
        doc_embs=doc_embs,
        venue_pad_id=venue_id_to_index[''],
        n_relations=n_relations,
        mode=trans_mode,
        device=device,
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, dataset_name, trans_mode, 'user_venue.pt'), map_location=device))
    model.eval()

    with open(f'{dataset_folder}/author_graph.json', 'r') as f:
        final_authors = json.load(f)

    doc_id_to_user = {}
    for a in tqdm(final_authors.keys()):
        for doc in final_authors[a]['wrote']:
            if doc in doc_id_to_user:
                doc_id_to_user[doc].append(a)
            else:
                doc_id_to_user[doc] = [a]


    print(dataset_folder)
    split = 'val'
    query_data = Indxr(os.path.join(dataset_folder, split, 'queries.jsonl'), key_id='id')
    user_run = get_bert_rerank(query_data, model, doc_id_to_user, user_id_to_index, device)

    qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
    # bert_ranx_run = Run(bert_run, name='BERT')
    bert_ranx_run = Run.from_file(os.path.join(runs_path, dataset_name, split, f'{model_save_name}.lz4'))
    bert_ranx_run.name = 'BERT'
    user_ranx_run = Run(user_run, name='USER')
    bm25_ranx_run = Run.from_file(os.path.join(dataset_folder, split, 'bm25_run.json'))
    bm25_ranx_run.name = 'BM25'


    bm25_user_params = optimize_fusion(
        qrels=qrels,
        runs=[bm25_ranx_run, user_ranx_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@100",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    bm25_bert_params = optimize_fusion(
        qrels=qrels,
        runs=[bm25_ranx_run, bert_ranx_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@100",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    bm25_user_run = fuse(
            runs=[bm25_ranx_run, user_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_user_params[0],
        )
    bm25_user_run.name = 'BM25 + USER'


    bm25_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_bert_params[0],
        )
    bm25_bert_run.name = 'BM25 + BERT'

    bm25_user_bert_params = optimize_fusion(
        qrels=qrels,
        runs=[bm25_ranx_run, bert_ranx_run, user_ranx_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@100",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    bm25_bert_user_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run, user_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_user_bert_params[0],
        )
    bm25_bert_user_run.name = 'BM25 + BERT + USER'

    models = [
        bm25_ranx_run,
        bert_ranx_run,
        user_ranx_run,
        bm25_user_run,
        bm25_bert_run,
        bm25_bert_user_run
    ]
    report = compare(
        qrels=qrels,
        runs=models,
        metrics=['map@100', 'mrr@10', 'ndcg@10'],
        max_p=0.01  # P-value threshold, 3 tests
    )
    print(report)

    split = 'test'
    query_data = Indxr(os.path.join(dataset_folder, split, 'queries.jsonl'), key_id='id')
    user_run = get_bert_rerank(query_data, model, doc_id_to_user, user_id_to_index, device)

    qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
    # bert_ranx_run = Run(bert_run, name='BERT')
    bert_ranx_run = Run.from_file(os.path.join(runs_path, dataset_name, split, f'{model_save_name}.lz4'))
    bert_ranx_run.name = 'BERT'
    user_ranx_run = Run(user_run, name='USER')
    bm25_ranx_run = Run.from_file(os.path.join(dataset_folder, split, 'bm25_run.json'))
    bm25_ranx_run.name = 'BM25'


    bm25_user_run = fuse(
            runs=[bm25_ranx_run, user_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_user_params[0],
        )
    bm25_user_run.name = 'BM25 + USER'

    bm25_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_bert_params[0],
        )
    bm25_bert_run.name = 'BM25 + BERT'

    bm25_bert_user_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run, user_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_user_bert_params[0],
        )
    bm25_bert_user_run.name = 'BM25 + BERT + USER'

    models = [
        bm25_ranx_run,
        bert_ranx_run,
        user_ranx_run,
        bm25_user_run,
        bm25_bert_run,
        bm25_bert_user_run
    ]
    report = compare(
        qrels=qrels,
        runs=models,
        metrics=['map@100', 'mrr@10', 'ndcg@10'],
        max_p=0.01  # P-value threshold, 3 tests
    )
    print(report)

if __name__ == "__main__":
    main()