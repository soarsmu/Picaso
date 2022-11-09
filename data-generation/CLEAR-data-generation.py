import pyarrow.parquet as pq
import numpy as np
import pandas as pd 
import pyarrow as pa 
from icecream import ic
import json
import time


def load_biker_data(filename):
    df = pd.read_csv(filename)
    df['title'] = df['title'].apply(lambda x: x.lower().replace(',', ''))
    return df[['title']]

def get_biker_id(df, query):
    try:
        return int(df[df['title']==query].index[0])
    except:
        ic("EXCEPTION", query)
        # exception_counter += 1

def read_pq(filename):
    table = pq.read_table(filename)
    return table.to_pandas()

def merge_parquet_data(fold_start=1, fold_end=14, output_file = 'generated-data/train'):
    trains = pd.DataFrame()
    triplets = pd.DataFrame()
    for i in range(fold_start, fold_end+1):
        ic(i)
        trains = pd.concat([trains, read_pq(f'train-{i}.parquet')])
        triplets = pd.concat([triplets, read_pq(f'train-{i}-triplets.parquet')])

    # table = pa.Table.from_pandas(trains)
    # pq.write_table(table, f'{output_file}-single.parquet')
    # table = pa.Table.from_pandas(triplets)
    # pq.write_table(table, f'{output_file}-triplets.parquet')

    return trains, triplets

def generate_annotation_corpus(train, validation, output_file):
    temp = pd.concat([train, validation])
    ic(train)
    ic(validation)
    ic(temp)

    corpus = {}
    for _, row in temp.iterrows():
        id = corpus.get(row['annotation'].lower().replace(',',''), len(corpus))
        corpus[row['annotation'].lower().replace(',','')] = id

    iCorpus = dict([(value, key) for key, value in corpus.items()])
    with open(output_file, 'w') as f:
        json.dump(iCorpus, f)

    ic(output_file)
    ic(len(corpus))
    # ic(len(set(temp['annotation'].to_list())))
    return corpus

def write_train_file(triplets, dictionary, biker_df, output_file):
    # output: {annotation_id: [[positive_ids], [negative_ids]]}
    start = time.time()
    res = {}
    ic(len(triplets))
    counter = 0
    for idx, row in triplets.iterrows():  #annotation, positive, negative
        ann = row['annotation'].lower().replace(',','')
        # id = dictionary[ann]
        id = dictionary.get(ann, False)
        if id==False and id != 0:
            ic(id)
            ic("### null ###", ann)
            test = list(dictionary.keys())[0]
            ic(test, dictionary[test])
            return None

        pos = res.get(id, [set(), set()])[0]
        neg = res.get(id, [set(), set()])[1]
        pos.add(get_biker_id(biker_df, row['positive']))
        neg.add(get_biker_id(biker_df, row['negative']))
        res[id] = [pos, neg]
        if counter%10000 == 0:
            ic(counter)
            ic(time.time()-start)
        counter += 1
    
    for key in res:
        val = res[key]
        val[0] = list(val[0])
        val[1] = list(val[1])

    ic("Writing Passage_dict.json") 
    with open(output_file, 'w') as f:
        json.dump(res, f)

def write_so_corpus(df, output_file):
    res = {}
    for idx, i in df.iterrows():
        res[int(idx)] = i['title'].lower().replace(',','')
    # ic(res)

    ic("Writing ", output_file) 
    with open(output_file, 'w') as f:
        json.dump(res, f)


exception_counter = 0
BIKER_FILE = '/app/CLEAR/data/BIKER_train.QApair.csv'
train, triplet = merge_parquet_data(1, 18)
ic(len(train))
ic(len(triplet))
validation, valid_triplet = merge_parquet_data(19, 20, 'generated-data/validation')
ic(len(validation))
ic(len(valid_triplet))

corpus = generate_annotation_corpus(train, validation, 'generated-data/CLEAR/Corpus.json')
# ic(corpus)

biker_df = load_biker_data(BIKER_FILE)
# write_so_corpus(biker_df, 'generated-data/CLEAR/SO_Corpus.json')

write_train_file(triplet, corpus, biker_df, 'generated-data/CLEAR/Passage_dict.json')
ic(valid_triplet)
write_train_file(valid_triplet, corpus, biker_df, 'generated-data/CLEAR/eval_dict.json')

# ic(exception_counter)