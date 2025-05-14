import random

from indxr import Indxr
from torch.utils.data import Dataset
from os.path import join

class MultiDomainDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        super(MultiDomainDataset, self).__init__()  
        self.data_dir = data_dir
        self.split = split

        self.split_dir = join(data_dir, split)

        self.corpus = Indxr(join(self.data_dir, 'collection.jsonl'), key_id='id')
        self.queries = Indxr(join(self.split_dir, 'queries.jsonl'), key_id='id')

    
    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['title'] if 'title' in query else query['text']
        rel_doc_id = random.choice(query['rel_doc_ids'])
        pos_doc_text = self.corpus.get(rel_doc_id)['title']

        retrieved_doc_ids = [q for q in query["bm25_doc_ids"] if q not in query['rel_doc_ids']]
        if retrieved_doc_ids:
            neg_doc_id = random.choice(retrieved_doc_ids)
            neg_doc_text = self.corpus.get(neg_doc_id)['title']
        else:
            neg_doc_text = '[SEP]'


        return {'query': query_text, 'pos_doc': pos_doc_text, 'neg_doc': neg_doc_text}

    def __len__(self):
        return len(self.queries)