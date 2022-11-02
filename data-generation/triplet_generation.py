from icecream import ic
import pandas as pd 
import args

parser = argparse.ArgumentParser()
parser.add_argument("--clear_data_path", type=int, required=True)
parser.add_argument("--deepapi_data_path", type=str, required=False)
parser.add_argument("--mularec_data_path", type=str, required=False)                    
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()


def process_clear_data():
    # 1. read clear dataset, get annotation and api in the post
    # return SO dataframe: id, annotation, api_list
    return None

def process_deepapi_data():
    # read hdf5 file and return annotation - api_seq in dataframe format
    return None

def find_relevant_post():
    # 1. input: javadoc annotation, target_api (we can use deepAPI dataset --> 7mio dataset)
    # 2. may need to set threshold (try: 0.75, 0.9, 1)
    return None

def build_triplet_for_datum(annotation, p=10, n=10):
    # find relevant post for annotation
    # return list of triplets, total: (p x n) rows
    return None

def build_triplet():
    # iterate all data and call build_triplet_for_datum
    return None

def write_dataset():
    # write dataframe to csv file (or parquet if csv is too big)
    return None