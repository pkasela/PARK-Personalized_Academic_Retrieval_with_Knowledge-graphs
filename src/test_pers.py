import torch
import os 
from model.pers_model import PersonalizationModel
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
@click.option('--device', default='cuda', help='Device')
@click.option('--aggregation_mode', default='mean', help='Aggregation mode')
@click.option('--embeddings_folder', default='embeddings', help='Embeddings folder')
@click.option('--model_name', default='all_minilm', help='Model name')
def main(dataset_folder, dataset_name, device, aggregation_mode, embeddings_folder, model_name):
    print(dataset_name, aggregation_mode)

    doc_embs = os.path.join(embeddings_folder,dataset_name, f'{model_name}.pt')
    doc_id_to_index = os.path.join(embeddings_folder, dataset_name, f'{model_name}.json')
    query_embs = os.path.join(embeddings_folder, dataset_name, f'{model_name}_query.pt')
    query_id_to_index = os.path.join(embeddings_folder, dataset_name, f'{model_name}_query.json')

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

    bm25_user_params = optimize_fusion(
        qrels=val_qrels,
        runs=[bm25_ranx_run, user_ranx_run],
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
        qrels=val_qrels,
        runs=[bm25_ranx_run, bert_ranx_run, user_ranx_run],
        norm="min-max",
        method="wsum",
        metric="map@100",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    user_bert_run = fuse(
            runs=[bm25_ranx_run, bert_ranx_run, user_ranx_run],
            norm="min-max",
            method="wsum",
            params=user_bert_params[0],
        )
    user_bert_run.name = 'BM25 + BERT + USER'

    models = [
        bm25_ranx_run,
        bert_ranx_run,
        user_ranx_run,
        bm25_user_run,
        bm25_bert_run,
        user_bert_run
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
            runs=[bm25_ranx_run, bert_ranx_run, user_ranx_run],
            norm="min-max",
            method="wsum",
            params=user_bert_params[0],
        )
    user_bert_run.name = 'BM25 + BERT + USER'

    models = [
        bm25_ranx_run,
        bert_ranx_run,
        user_ranx_run,
        bm25_user_run,
        bm25_bert_run,
        user_bert_run
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
