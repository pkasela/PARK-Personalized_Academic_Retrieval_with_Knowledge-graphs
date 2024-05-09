from torch.utils.data import Dataset
from tqdm import tqdm
import random
import json
import subprocess


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_jsonl(path, verbose=True):
    with open(path) as f:
        pbar = tqdm(f, total=file_len(path), desc=f'Loading {path}') if verbose else f
        data = [json.loads(line) for line in pbar]
    return data


class GraphDataLoader(Dataset):
    def __init__(self, queries, collection, authors, out_refs, user_docs=20, cited_docs=10):
        
        self.queries = queries
        self.collection = { c['id']: c for c in collection }
        self.authors    = { a['id']: a for a in authors }
        self.out_refs   = { d['doc_id']: d for d in out_refs }

        self.user_docs = user_docs
        self.cited_docs = cited_docs

    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['text']
        
        rel_doc_id, pos_doc_info, neg_doc_id, neg_doc_info = self.get_positive_and_negative(query)
        
        user_id, user_docs = self.get_user_info(query['user_id'])

        return {
            'query': query_text,
            'pos_doc': pos_doc_info['text'],
            'pos_doc_venue': pos_doc_info['venue_id'],
            'pos_cited': pos_doc_info['cited'],
            'neg_doc': neg_doc_info['text'],
            'neg_doc_venue': neg_doc_info['venue_id'],
            'neg_cited': neg_doc_info['cited'],
            'user_id': user_id,
            'user_docs': user_docs   
        }

    def get_positive_and_negative(self, query):
        rel_doc_id = random.choice(query['rel_doc_ids'])
        pos_doc_info = self.get_doc_info(rel_doc_id)

        retrieved_doc_ids = [q for q in query["bm25_doc_ids"] if q not in query['rel_doc_ids']]

        neg_doc_id = random.choice(retrieved_doc_ids)
        neg_doc_info = self.get_doc_info(neg_doc_id)
            
        return rel_doc_id, pos_doc_info, neg_doc_id, neg_doc_info
    
    def get_user_info(self, user_id):
        author_infos = self.authors[user_id]
        author_docs = [self.collection[doc['doc_id']]['title'] for doc in random.sample(author_infos['docs'], self.user_docs)]

        return user_id, author_docs
    
    def get_doc_info(self, doc_id):
        doc_infos = self.collection[doc_id]
        doc_text = doc_infos['title']
        if doc_infos['conference_instance_id'] == "":
            if doc_infos['conference_series_id'] == "":
                venue_id = doc_infos['journal_id']
            else:
                venue_id = doc_infos['conference_series_id']
        else:
            venue_id = doc_infos['conference_instance_id']

        doc_references = [self.collection[doc]['title'] for doc in self.out_refs[doc_id]['out_refs'] if doc in self.collection]
        if len(doc_references) > self.cited_docs:
            doc_references = random.sample(doc_references, self.cited_docs) 
        if len(doc_references) < self.cited_docs:
            doc_references = doc_references + ['[SEP]' for _ in range(self.cited_docs - len(doc_references))]
        

        return {
            'text' : doc_text,
            'venue_id': venue_id,
            'cited': doc_references
        }
    
    def __len__(self):
        return len(self.queries)

def collate_fn(batch):
    return {
            'query': [x.get('query') for x in batch],
            'pos_doc': [x.get('pos_doc') for x in batch],
            'pos_doc_venue': [x.get('pos_doc_venue') for x in batch],
            'pos_cited': [x.get('pos_cited') for x in batch],
            'neg_doc': [x.get('neg_doc') for x in batch],
            'neg_doc_venue': [x.get('neg_doc_venue') for x in batch],
            'neg_cited': [x.get('neg_cited') for x in batch],
            'user_id': [x.get('user_id') for x in batch],
            'user_docs': [x.get('user_docs') for x in batch]   
        }
