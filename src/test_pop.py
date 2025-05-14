import torch
import os 
from dataloader.dataloader import read_jsonl
from tqdm import tqdm
from ranx import Run, Qrels, compare, fuse, optimize_fusion
import click

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

@click.command()
@click.option('--dataset_folder', default='psychology', help='Dataset folder')
@click.option('--dataset_name', default='psychology', help='Dataset name')
@click.option('--model_save_name', default='all_minilm', help='Model name')
@click.option('--runs_path', default='runs', help='Runs path')
def main(dataset_folder, dataset_name, model_save_name, runs_path):
    print(dataset_name)

    
    split = 'val'

    val_bert_run = Run.from_file(os.path.join(runs_path, dataset_name, split ,f'{model_save_name}.lz4'))
    val_bert_run.name = 'BERT'
    val_qrels = Qrels.from_file(os.path.join(dataset_folder, split, 'qrels.json'))
    bert_ranx_run = val_bert_run

    bm25_ranx_run = Run.from_file(os.path.join(dataset_folder, split, 'bm25_run.json'))
    bm25_ranx_run.name = 'BM25'
    pop_ranx_run = Run.from_file(os.path.join(dataset_folder, f'Pop_{split}.json'))
    pop_ranx_run.name = 'POP'

    bm25_pop_params = optimize_fusion(
        qrels=val_qrels,
        runs=[bm25_ranx_run, pop_ranx_run],
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

    bm25_pop_run = fuse(
            runs=[bm25_ranx_run, pop_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_pop_params[0],
        )
    bm25_pop_run.name = 'BM25 + POP'


    bm25_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_bert_params[0],
        )
    bm25_bert_run.name = 'BM25 + BERT'

    pop_bert_params = optimize_fusion(
        qrels=val_qrels,
        runs=[bm25_ranx_run, bert_ranx_run, pop_ranx_run],
        norm="min-max",
        method="wsum",
        metric="map@100",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    pop_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run, pop_ranx_run],
            norm="min-max",
            method="wsum",
            params=pop_bert_params[0],
        )
    pop_bert_run.name = 'BM25 + BERT + POP'

    models = [
        bm25_ranx_run,
        bert_ranx_run,
        pop_ranx_run,
        bm25_pop_run,
        bm25_bert_run,
        pop_bert_run
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
    pop_ranx_run = Run.from_file(os.path.join(dataset_folder, f'Pop_{split}.json'))
    pop_ranx_run.name = 'POP'


    bm25_pop_run = fuse(
            runs=[bm25_ranx_run, pop_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_pop_params[0],
        )
    bm25_pop_run.name = 'BM25 + POP'

    bm25_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run],
            norm="min-max",
            method="wsum",
            params=bm25_bert_params[0],
        )
    bm25_bert_run.name = 'BM25 + BERT'

    pop_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run, pop_ranx_run],
            norm="min-max",
            method="wsum",
            params=pop_bert_params[0],
        )
    pop_bert_run.name = 'BM25 + BERT + POP'

    models = [
        bm25_ranx_run,
        bert_ranx_run,
        pop_ranx_run,
        bm25_pop_run,
        bm25_bert_run,
        pop_bert_run
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