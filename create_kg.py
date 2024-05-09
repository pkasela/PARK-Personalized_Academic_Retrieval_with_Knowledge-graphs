import json
import os
import subprocess
from tqdm import tqdm
from datetime import datetime
from time import mktime

def year_to_timestamp(year):
    try:
        return (
            t
            if (
                t := int(mktime(datetime.strptime(str(year), "%Y").timetuple()))
            )
            > 0
            else 0
        )
    except:
        print(year)

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

def write_jsonl(path, data, verbose=True):
    with open(path, 'w') as f:
        pbar = tqdm(data, desc=f'Writing {path}') if verbose else data
        for d in pbar:
            json.dump(d, f)
            f.write('\n')

# remove the authors with less than 20 papers:
dataset_folder = 'computer_science'
output_folder = f'cleaned_{dataset_folder}'
split_year = 2017
test_time = year_to_timestamp(split_year)

authors = read_jsonl(os.path.join(dataset_folder, 'authors.jsonl'))

collection = read_jsonl(os.path.join(dataset_folder, 'collection.jsonl'))
# doc_ids = set([doc['id'] for doc in collection])
venue_ids = []
for doc_infos in tqdm(collection):
    if doc_infos['conference_instance_id'] == "":
        if doc_infos['conference_series_id'] == "":
            venue_id = doc_infos['journal_id']
        else:
            venue_id = doc_infos['conference_series_id']
    else:
        venue_id = doc_infos['conference_instance_id']
    venue_ids.append(venue_id)
train_queries = read_jsonl(os.path.join(dataset_folder, 'train/queries.jsonl'))
val_queries   = read_jsonl(os.path.join(dataset_folder, 'val/queries.jsonl'))
test_queries  = read_jsonl(os.path.join(dataset_folder, 'test/queries.jsonl'))

user_ids = set([query['user_id'] for query in tqdm(train_queries + val_queries + test_queries)])

good_authors = [a for a in tqdm(authors) if a['id'] in user_ids]

authors_wrote_paper = [(a['id'], 'wrote', doc['doc_id']) for a in tqdm(good_authors) for doc in a['docs']]

os.makedirs(output_folder)
write_jsonl(os.path.join(output_folder, 'authors.jsonl'), good_authors)











