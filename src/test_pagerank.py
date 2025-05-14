import networkx as nx
import json
import os 
from dataloader.dataloader import read_jsonl
from tqdm import tqdm
from ranx import Run, Qrels, compare, fuse, optimize_fusion
import click


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
device = 'cuda'
aggregation_mode = 'mean'

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

@click.command()
@click.option('--dataset_folder', default='physics', help='Dataset folder')
@click.option('--dataset_name', default='physics', help='Dataset name')
@click.option('--out_ref_file', default=None, help='Path to out_refs.jsonl')
@click.option('--collection_file', default=None, help='Path to collection.jsonl')
@click.option('--model_save_name', default='all_minilm', help='Model name')
@click.option('--runs_path', default='runs', help='Runs path')
def main(dataset_folder, dataset_name, out_ref_file, collection_file, model_save_name, runs_path):
    if out_ref_file is None:
        out_ref_file = f'{dataset_folder}/out_refs.jsonl'
    if collection_file is None:
        collection_file = f'{dataset_folder}/collection.jsonl'

    # Load the out_refs and collection files
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

    print(dataset_name)

    split = 'val'
    queries = read_jsonl(os.path.join(dataset_folder, split, 'queries.jsonl'))

    val_bert_run = Run.from_file(os.path.join(runs_path, dataset_name, split ,f'{model_save_name}.lz4'))
    val_bert_run.name = 'BERT'
    val_qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
    bert_ranx_run = val_bert_run # Run(val_bert_run, name='BERT')

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

    test_bert_run = Run.from_file(os.path.join(runs_path, dataset_name, split ,f'{model_save_name}.lz4'))
    test_bert_run.name = 'BERT'
    test_qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
    bert_ranx_run = test_bert_run # Run(test_bert_run, name='BERT')

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

if __name__ == '__main__':
    main()