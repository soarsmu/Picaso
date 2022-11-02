import sys
import json
import torch 
import tables
import pickle
import random
import numpy as np
from icecream import ic
from helper import PAD_ID, SOS_ID, EOS_ID, UNK_ID
import re

use_cuda = torch.cuda.is_available()

def load_dict(filename):
    return json.loads(open(filename, "r", encoding="utf-8").readline())
    
class APIDataset():
    def __init__(self, desc_file, api_file, api_vocab_file, desc_vocab_file):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        api_table = tables.open_file(api_file)
        self.api_data = api_table.get_node('/phrases')[:].astype(np.long)
        self.api_index = api_table.get_node('/indices')[:]
        
        desc_table = tables.open_file(desc_file)
        self.desc_data = desc_table.get_node('/phrases')[:].astype(np.long)
        self.desc_index = desc_table.get_node('/indices')[:]
        
        assert self.api_index.shape[0] == self.desc_index.shape[0], "inconsistent number of API sequences and NL descriptions!"
        self.data_len = self.api_index.shape[0]
        ic(self.data_len)

        # vocab_api = load_dict(input_dir+'vocab.apiseq.pkl')
        vocab_api = load_dict(api_vocab_file)
        self.ivocab_api = {v: k for k, v in vocab_api.items()}

        # vocab_desc = load_dict(input_dir+'vocab.desc.pkl')
        vocab_desc = load_dict(desc_vocab_file)
        self.ivocab_desc = {v: k for k, v in vocab_desc.items()}
        
    def list2array(self, L, max_len, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        arr = np.zeros(max_len, dtype=dtype)+pad_idx
        for i, v in enumerate(L): arr[i] = v
        return arr
    
    def get_item(self, offset):
        pos, api_len = self.api_index[offset]['pos'], self.api_index[offset]['length']
        # api_len = min(int(api_len),self.max_seq_len-2) # real length of sequences
        api = self.api_data[pos: pos + api_len].tolist()
        
        pos, desc_len = self.desc_index[offset]['pos'], self.desc_index[offset]['length']
        # desc_len=min(int(desc_len),self.max_seq_len-2) # get real seq len
        desc= self.desc_data[pos: pos+ desc_len].tolist()

        # convert id to word with dictionary
        api_token = "###sep###".join([self.ivocab_api.get(x, UNK_ID) for x in api])
        api_token = api_token.replace("<init>", "INIT")
        api_token = re.sub(r'\<.*?\>', '', api_token)
        api_token = api_token.replace('>', '')
        desc_token = ' '.join([self.ivocab_desc.get(x, UNK_ID) for x in desc])

        return api_token, desc_token
    
if __name__ == '__main__':
    input_dir='/app/deepAPI/deepapi_dataset/'
    VALID_FILE_API=input_dir+'test.apiseq.h5'
    VALID_FILE_DESC=input_dir+'test.desc.h5'
    API_VOCAB_FILE=input_dir+'vocab.apiseq.json'
    DESC_VOCAB_FILE=input_dir+'vocab.desc.json'

    data = APIDataset(VALID_FILE_DESC, VALID_FILE_API, API_VOCAB_FILE, DESC_VOCAB_FILE)
    ic(data.data_len)
    
    for i in range(data.data_len):
        if i > 5:
            break
        ic(i, data.get_item(i))