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

NUCLEOTIDS = ['A', 'T', 'G', 'C']
GENOMES_MAX_LEN = 100

def numerize_genome_str(x):
    encoded_x = np.zeros(len(NUCLEOTIDS) * GENOMES_MAX_LEN)
    for i in range(len(x)):
        for j in range(len(NUCLEOTIDS)):
            if x[i] == NUCLEOTIDS[j]:
                encoded_x[i * len(NUCLEOTIDS) + j] = 1.0
                break
    
    return encoded_x


if __name__ == "__main__":
    x = 'ATACAGACCATTGTTATATTCATATATGTTAAGATTAAGTTTCTTAAGTGACATATGAACGATGTCATACACTTCTGCAT'
    print(numerize_genome_str(x))