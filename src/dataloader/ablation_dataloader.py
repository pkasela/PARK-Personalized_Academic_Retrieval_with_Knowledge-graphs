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

class OnlyAuthorGraphDataLoader(Dataset):
    def __init__(
            self, 
            author_graph_file,
            user_id_to_index,
            doc_id_to_index
        ):
        with open(author_graph_file, 'r') as f:
            self.data = json.load(f)
        self.authors = list(self.data.keys())
        self.user_id_to_index = user_id_to_index
        self.doc_id_to_index = doc_id_to_index

    def __getitem__(self, idx):
        author = self.authors[idx]
        author_data = self.data[author]

        author_wrote = self.doc_id_to_index[random.sample(author_data['wrote'], k=1)[0]]
        try:
            author_cited = self.doc_id_to_index[random.sample(author_data['cited'], k=1)[0]]
        except:
            author_cited = self.doc_id_to_index[random.sample(author_data['wrote'], k=1)[0]]
        author_coauthor = self.user_id_to_index[random.sample(author_data['co_author'], k=1)[0]]
        
        return {
            'user_id': self.user_id_to_index[author],
            'wrote': author_wrote,
            'cited': author_cited,
            'coauthor': author_coauthor,
        }
    
    def __len__(self):
        return len(self.authors)

def only_author_collate_fn(batch):
    return {
            'user_id': [x.get('user_id') for x in batch],
            'wrote': [x.get('wrote') for x in batch],
            'cited': [x.get('cited') for x in batch],
            'coauthor': [x.get('coauthor') for x in batch],
        }

class UserVenueAuthorGraphDataLoader(Dataset):
    def __init__(
            self, 
            author_graph_file,
            user_id_to_index,
            venue_id_to_index,
            doc_id_to_index
        ):
        with open(author_graph_file, 'r') as f:
            self.data = json.load(f)
        self.authors = list(self.data.keys())
        self.user_id_to_index = user_id_to_index
        self.venue_id_to_index = venue_id_to_index
        self.doc_id_to_index = doc_id_to_index

    def __getitem__(self, idx):
        author = self.authors[idx]
        author_data = self.data[author]

        author_wrote = self.doc_id_to_index[random.sample(author_data['wrote'], k=1)[0]]
        try:
            author_cited = self.doc_id_to_index[random.sample(author_data['cited'], k=1)[0]]
        except:
            author_cited = self.doc_id_to_index[random.sample(author_data['wrote'], k=1)[0]]
        author_coauthor = self.user_id_to_index[random.sample(author_data['co_author'], k=1)[0]]
        author_venue = self.venue_id_to_index[random.sample(author_data['venue'], k=1)[0]]

        return {
            'user_id': self.user_id_to_index[author],
            'wrote': author_wrote,
            'cited': author_cited,
            'coauthor': author_coauthor,
            'venue': author_venue,
        }
    
    def __len__(self):
        return len(self.authors)

def user_venue_author_collate_fn(batch):
    return {
            'user_id': [x.get('user_id') for x in batch],
            'wrote': [x.get('wrote') for x in batch],
            'cited': [x.get('cited') for x in batch],
            'coauthor': [x.get('coauthor') for x in batch],
            'venue': [x.get('venue') for x in batch],
        }
