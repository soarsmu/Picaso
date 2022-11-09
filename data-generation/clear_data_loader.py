import sys
from icecream import ic
from helper import PAD_ID, SOS_ID, EOS_ID, UNK_ID
import re
import ast
import pandas as pd

def load_dict(filename):
    return json.loads(open(filename, "r", encoding="utf-8").readline())
    
class ClearAPIDataset():
    def __init__(self, biker_train_file):
        self.data = pd.read_csv(biker_train_file, index_col=0)
        # self.data = self.data.drop(columns='Unnamed: 0')
        self.data['answer'] = self.data['answer'].apply(lambda x: ast.literal_eval(x.lower()))
        self.data['processed_answer'] = self.data['answer'].apply(lambda x: ['.'.join(y.split('.')[-2:]) for y in x])
        self.data['answer_set'] = self.data['processed_answer'].apply(lambda x: set(x))
        self.data_len = len(self.data)
    
if __name__ == '__main__':
    BIKER_TRAINING_FILE='/app/CLEAR/data/BIKER_train.QApair.csv'
    biker_data = ClearAPIDataset(BIKER_TRAINING_FILE)
    # ic(biker_data.data)
