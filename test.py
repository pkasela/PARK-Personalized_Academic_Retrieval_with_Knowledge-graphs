import os
import json
import torch

from src.dataloader.dataloader import read_jsonl 
from src.model.model import GraphBiEncoder
from tqdm import tqdm
from indxr import Indxr
from ranx import Run, Qrels, compare, fuse, optimize_fusion


def get_bert_rerank(data, model, doc_embedding, id_to_index):
    acceptable_authors = author_id_to_index.keys()
    bert_run = {}
    user_run = {}
    model.eval()
    for query in tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder([query['text']])

        bm25_docs = query['bm25_doc_ids']
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[query['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}

        if query['user_id'] not in acceptable_authors:
            user_run[query['id']] =  {doc_id: 0. for i, doc_id in enumerate(bm25_docs)}
        else:
            u_embedding = model.user_embeddings([query['user_id']])
            # wrote_relation = model.relation_embedding(torch.tensor(1).to('cuda'))
            cited_relation = model.relation_embedding(torch.tensor(0).to('cuda'))
            u_embedding = u_embedding + cited_relation
            user_scores = torch.einsum('xy, ly -> x', d_embeddings, u_embedding)
            user_run[query['id']] = {doc_id: user_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
                

    return bert_run, user_run


datafolder = 'computer_science'
split = 'train'


train_queries    = read_jsonl(os.path.join(datafolder, split, 'queries.jsonl'))
collection = read_jsonl(os.path.join(datafolder, 'collection.jsonl'))
authors    = read_jsonl(os.path.join(datafolder, 'authors.jsonl'))
out_refs   = read_jsonl(os.path.join(datafolder, 'out_refs.jsonl'))
authors_ids = list(set([q['user_id'] for q in train_queries]))
author_id_to_index = {a_id: i for i, a_id in enumerate(authors_ids)}

doc_id_to_authors = {}
for author in tqdm(authors):
    for doc in author['docs']:
        if doc['doc_id'] in doc_id_to_authors:
            doc_id_to_authors[doc['doc_id']].append(author['id'])
        else: 
            doc_id_to_authors[doc['doc_id']] = [author['id']]

venue_ids = []
for doc in collection:
    if doc['conference_instance_id'] == "":
        if doc['conference_series_id'] == "":
            venue_id = doc['journal_id']
        else:
            venue_id = doc['conference_series_id']
    else:
        venue_id = doc['conference_instance_id']

    venue_ids.append(venue_id)
venue_ids = list(set(venue_ids))
venue_id_to_index = {v_id: i for i, v_id in enumerate(venue_ids)}
venue_id_to_index['-1'] = len(venue_ids)


model = GraphBiEncoder(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
    author_to_index=author_id_to_index,
    venue_to_index=venue_id_to_index,
    device='cuda'
)
# model.load(f'../multidomain/models/{datafolder}/all_minilm.pt')
model.load_state_dict(torch.load('models/trained_model.pt'))
model.eval()

doc_embedding = torch.load(os.path.join('embeddings', 'all_minilm.pt')).to('cuda')
with open(os.path.join('embeddings', 'all_minilm.json'), 'r') as f:
    id_to_index = json.load(f)

split = 'val'
query_data = Indxr(os.path.join(datafolder, split, 'queries.jsonl'), key_id='id')
bert_run, user_run = get_bert_rerank(query_data, model, doc_embedding, id_to_index)

qrels = Qrels.from_file(os.path.join(datafolder, split, 'qrels.json'))
bert_ranx_run = Run(bert_run, name='BERT')
# bert_ranx_run = Run.from_file(f'../multidomain/runs/computer_science/{split}/all_minilm.lz4')
# bert_ranx_run.name = 'BERT'
user_ranx_run = Run(user_run, name='USER')
bm25_ranx_run = Run.from_file(os.path.join(datafolder, split, 'bm25_run.json'))
bm25_ranx_run.name = 'BM25'


bm25_user_params = optimize_fusion(
    qrels=qrels,
    runs=[bm25_ranx_run, user_ranx_run],
    norm="min-max",
    method="wsum",
    metric="map@100",  # The metric to maximize during optimization
    return_optimization_report=True
)

bm25_bert_params = optimize_fusion(
    qrels=qrels,
    runs=[bm25_ranx_run, bert_ranx_run],
    norm="min-max",
    method="wsum",
    metric="map@100",  # The metric to maximize during optimization
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

user_bert_params = optimize_fusion(
    qrels=qrels,
    runs=[bm25_user_run, bm25_bert_run],
    norm="min-max",
    method="wsum",
    metric="map@100",  # The metric to maximize during optimization
    return_optimization_report=True
)

user_bert_run = fuse(
        runs=[bm25_user_run, bm25_bert_run],
        norm="min-max",
        method="wsum",
        params=user_bert_params[0],
    )
user_bert_run.name = '(BM25 + BERT) + (BM25 + USER)'

models = [
    bm25_ranx_run,
    bert_ranx_run,
    user_ranx_run,
    bm25_user_run,
    bm25_bert_run,
    user_bert_run
]
report = compare(
    qrels=qrels,
    runs=models,
    metrics=['map@100', 'mrr@10', 'ndcg@10'],
    max_p=0.01  # P-value threshold, 3 tests
)
print(report)

split = 'test'
query_data = Indxr(os.path.join(datafolder, split, 'queries.jsonl'), key_id='id')
bert_run, user_run = get_bert_rerank(query_data, model, doc_embedding, id_to_index)

qrels = Qrels.from_file(os.path.join(datafolder, split, 'qrels.json'))
bert_ranx_run = Run(bert_run, name='BERT')
# bert_ranx_run = Run.from_file(f'../multidomain/runs/computer_science/{split}/all_minilm.lz4')
# bert_ranx_run.name = 'BERT'
user_ranx_run = Run(user_run, name='USER')
bm25_ranx_run = Run.from_file(os.path.join(datafolder, split, 'bm25_run.json'))
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

user_bert_run = fuse(
        runs=[bm25_user_run, bm25_bert_run],
        norm="min-max",
        method="wsum",
        params=user_bert_params[0],
    )
user_bert_run.name = '(BM25 + BERT) + (BM25 + USER)'

models = [
    bm25_ranx_run,
    bert_ranx_run,
    user_ranx_run,
    bm25_user_run,
    bm25_bert_run,
    user_bert_run
]
report = compare(
    qrels=qrels,
    runs=models,
    metrics=['map@100', 'mrr@10', 'ndcg@10'],
    max_p=0.01  # P-value threshold, 3 tests
)
print(report)