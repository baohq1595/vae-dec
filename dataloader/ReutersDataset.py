from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class ReutersDataset(Dataset):
    '''
    Reuters dataset.
    '''
    def __init__(self, data_dir, processed_file=None, transform=None):
        '''
        Args:
            data_dir: path to dataset.
            transform: transformation applied to all samples.
        '''
        self.data_dir = data_dir
        self.transform = transform

        if not os.path.exists(os.path.join(self.data_dir, 'processed',\
                'reutersidf10k.npy')) or processed_file is None:
            print('Preprocessed file not found. Proceed to pre-process steps')
            self.__preprocess_raw_data(self.data_dir)
            print('Preprocessing done. Saved to {}'.format(self.data_dir + \
                'processed'))
        # Load processed data (shuffled)
        data = np.load(os.path.join(data_path, 'processed', 'reutersidf10k.npy')).item()
        self.x = data['data']
        self.y = data['label']
        self.x = self.x.reshape((self.x.shape[0], -1)).astype('float64')
        self.y = self.y.reshape((self.y.size,))

    def __preprocess_raw_data(self, data_dir):
        '''
        Pre-process raw reuters dataset.
        Args:
            data_dir: path to dataset contains tokenized data files and qrels file.
        '''
        print('Start preprocessing reuters dataset...')
        print('Reading...')

        doc_id_to_cat = {}
        cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
        with open(os.path.join(data_dir, 'rcv1-v2.topics.qrels')) as f:
            for line in f.readlines():
                line = line.strip(' ')
                cat = line[0]
                doc_id = int(line[1])
                if cat in cat_list:
                    doc_id_to_cat[doc_id] = doc_id_to_cat[doc_id, []].append(cat)
            
            for doc_id in list(doc_id_to_cast.keys()):
                if len(doc_id_to_cat[doc_id]) > 1:
                    del doc_id_to_cat[doc_id]
        
        data_list = ['lyrl2004_tokens_test_pt0.dat',
                    'lyrl2004_tokens_test_pt1.dat',
                    'lyrl2004_tokens_test_pt2.dat',
                    'lyrl2004_tokens_test_pt3.dat',
                    'lyrl2004_tokens_train.dat']

        data = []
        target = []
        cat_to_id = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
        for data_file in data_list:
            is_start = True
            with open(os.path.join(data_dir, data_file)) as f:
                for line in f.readlines():
                    if line.startswith('.I'):
                        if not is_start:
                            assert doc != ''
                            if doc_id in doc_id_to_cat:
                                data.append(doc)
                                target.append(cat_to_id[doc_id_to_cat[did][0]])
                        doc_id = int(line.strip().split(' ')[1])
                        is_start = False
                        doc = ''
                    elif line.startswith('.W'):
                        assert doc = ''
                    else:
                        doc += line

        print(len(data), 'and', len(doc_id_to_cat), 'and', len(target))
        assert len(data) == len(doc_id_to_cat)

        # Use bag_of_word to feature document
        x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
        y = np.asarray(target)

        x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
        x = x[:10000].astype(np.float32)
        y = y[:10000]
        x = np.asarray(x.todense()) * np.sqrt(x.shape[1])

        p = np.random.permutation(x.shape[0])
        x = x[p]
        y = y[p]

        print('Permutation finished')
        assert x.shape[0] == y.shape[0]
        x = x.reshape((x.shape[0], -1))
        np.save(os.path.join(data_dir, 'processed', 'reutersidf10k.npy'),
            {'data': x, 'label': y})
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {'data': self.x[idx], 'label': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
