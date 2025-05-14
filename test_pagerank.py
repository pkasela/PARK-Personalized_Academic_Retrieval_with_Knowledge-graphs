import networkx as nx
import json
import torch
from torch import nn
import os 
from pers_model import PersonalizationModel
from dataloader import read_jsonl
from tqdm import tqdm
from ranx import Run, Qrels, compare, fuse, optimize_fusion

def get_user_rerank(data, model, top_k=1000):
    user_run = {}
    bert_run = {}
    for query in tqdm(data, total=len(data)):
        bm25_docs = query['bm25_doc_ids']
        
        batch = {
            'query_id': [query['id']], 
            'pos_doc_id': query['bm25_doc_ids'], 
            'user_doc_id': query['user_doc_ids']
        }
        user_embs, d_embeddings, query_embedding = model(batch)

        user_scores = torch.einsum('xy, ly -> x', d_embeddings, user_embs)
        user_run[query['id']] = {doc_id: user_scores[i].item() for i, doc_id in enumerate(bm25_docs[:top_k])}                
        
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, query_embedding)
        bert_run[query['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs[:top_k])}                
        
    return user_run, bert_run


def get_pagerank_rerank(data, pr, top_k=1000):
    pr_run = {}
    for query in tqdm(data, total=len(data)):
        bm25_docs = query['bm25_doc_ids']
        
        batch = {
            'query_id': [query['id']], 
            'pos_doc_id': query['bm25_doc_ids'], 
            'user_doc_id': query['user_doc_ids']
        }
        pr_run[query['id']] = {doc_id: pr[doc_id] for doc_id in bm25_docs[:top_k]}
        
    return pr_run

dataset_folder = 'physics'
dataset_name = 'physics'
out_ref_file = f'{dataset_folder}/out_refs.jsonl'
collection_file = f'{dataset_folder}/collection.jsonl'

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

out_ref = read_jsonl(out_ref_file)

collection = read_jsonl(collection_file)


graph_ids = [c['id'] for c in collection]

G = nx.DiGraph()
G.add_nodes_from(graph_ids)

edges = []
for refs in out_ref:
    for ref in refs['out_refs']:
        edges.append((refs['doc_id'], ref))

G.add_edges_from(edges)
pr = nx.pagerank(G)


device = 'cuda'
aggregation_mode = 'mean'

print(dataset_name)

doc_embs = os.path.join('embeddings',dataset_name, 'all_minilm.pt')
doc_id_to_index = os.path.join('embeddings', dataset_name, 'all_minilm.json')
query_embs = os.path.join('embeddings', dataset_name, 'all_minilm_query.pt')
query_id_to_index = os.path.join('embeddings', dataset_name, 'all_minilm_query.json')

model = PersonalizationModel(
    doc_embs, 
    doc_id_to_index, 
    query_embs, 
    query_id_to_index, 
    aggregation_mode, 
    device
)

split = 'val'
queries = read_jsonl(os.path.join(dataset_folder, split, 'queries.jsonl'))

val_user_run, val_bert_run = get_user_rerank(queries, model, top_k=1000)
val_qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
bert_ranx_run = Run(val_bert_run, name='BERT')
user_ranx_run = Run(val_user_run, name='USER')

bm25_ranx_run = Run.from_file(os.path.join(dataset_folder, split, 'bm25_run.json'))
bm25_ranx_run.name = 'BM25'
pr_run = get_pagerank_rerank(queries, pr, top_k=1000)
pr_ranx_run = Run(pr_run)
pr_ranx_run.name = 'PageRank'

bm25_pr_params = optimize_fusion(
    qrels=val_qrels,
    runs=[bm25_ranx_run, pr_ranx_run],
    norm="min-max",
    method="wsum",
    metric="map@100",  # The metric to maximize during optimization
    return_optimization_report=True
)
bm25_bert_params = optimize_fusion(
    qrels=val_qrels,
    runs=[bm25_ranx_run, bert_ranx_run],
    norm="min-max",
    method="wsum",
    metric="map@100",  # The metric to maximize during optimization
    return_optimization_report=True
)

bm25_pr_run = fuse(
        runs=[bm25_ranx_run, pr_ranx_run],
        norm="min-max",
        method="wsum",
        params=bm25_pr_params[0],
    )
bm25_pr_run.name = 'BM25 + PageRank'


bm25_bert_run = fuse(
        runs=[bm25_ranx_run, bert_ranx_run],
        norm="min-max",
        method="wsum",
        params=bm25_bert_params[0],
    )
bm25_bert_run.name = 'BM25 + BERT'

pr_bert_params = optimize_fusion(
    qrels=val_qrels,
    runs=[bm25_ranx_run, bert_ranx_run, pr_ranx_run],
    norm="min-max",
    method="wsum",
    metric="map@100",  # The metric to maximize during optimization
    return_optimization_report=True
)

pr_bert_run = fuse(
        runs=[bm25_ranx_run, bert_ranx_run, pr_ranx_run],
        norm="min-max",
        method="wsum",
        params=pr_bert_params[0],
    )
pr_bert_run.name = 'BM25 + BERT + PageRank'

models = [
    bm25_ranx_run,
    bert_ranx_run,
    pr_ranx_run,
    bm25_pr_run,
    bm25_bert_run,
    pr_bert_run
]
report = compare(
    qrels=val_qrels,
    runs=models,
    metrics=['map@100', 'mrr@10', 'ndcg@10'],
    max_p=0.01  # P-value threshold, 3 tests
)
print(report)

split = 'test'
test_queries = read_jsonl(os.path.join(dataset_folder, split, 'queries.jsonl'))

test_user_run, test_bert_run = get_user_rerank(test_queries, model, top_k=1000)
test_qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
bert_ranx_run = Run(test_bert_run, name='BERT')
user_ranx_run = Run(test_user_run, name='USER')

bm25_ranx_run = Run.from_file(os.path.join(dataset_folder, split, 'bm25_run.json'))
bm25_ranx_run.name = 'BM25'
pr_run = get_pagerank_rerank(test_queries, pr, top_k=1000)
pr_ranx_run = Run(pr_run) #Run.from_file(os.path.join(dataset_folder, f'pr_{split}.json'))
pr_ranx_run.name = 'pr'


bm25_pr_run = fuse(
        runs=[bm25_ranx_run, pr_ranx_run],
        norm="min-max",
        method="wsum",
        params=bm25_pr_params[0],
    )
bm25_pr_run.name = 'BM25 + pr'

bm25_bert_run = fuse(
        runs=[bm25_ranx_run, bert_ranx_run],
        norm="min-max",
        method="wsum",
        params=bm25_bert_params[0],
    )
bm25_bert_run.name = 'BM25 + BERT'

pr_bert_run = fuse(
        runs=[bm25_ranx_run, bert_ranx_run, pr_ranx_run],
        norm="min-max",
        method="wsum",
        params=pr_bert_params[0],
    )
pr_bert_run.name = 'BM25 + BERT + pr'

models = [
    bm25_ranx_run,
    bert_ranx_run,
    pr_ranx_run,
    bm25_pr_run,
    bm25_bert_run,
    pr_bert_run
]
report = compare(
    qrels=test_qrels,
    runs=models,
    metrics=['map@100', 'mrr@10', 'ndcg@10'],
    max_p=0.01  # P-value threshold, 3 tests
)
print(report)