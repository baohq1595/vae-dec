import os
import torch
import re, random
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append('.')
from dataloader.utils import generate_k_mer_corpus, ensure_gene_length

class GenomeDataset(Dataset):
    '''
    Metagenomics dataset for reading simulated data in fasta format (.fna)
    '''

    HASH_PATTERN = r'\([a-f0-9]{40}\)'
    def __init__(self, fna_file, feature_type='bow', k_mer=4, overlap_k_mer=True, transform=None, is_normalize=False):
        '''
        Args:
            k_mer: number of nucleotid to combine into a word.
            overlap_k_mer: True to extract overlapping k_mer from a genome string. False otherwise.
            fna_file: path to fna file (fasta format).
            transform: transformation applied to all samples.
        '''
        assert os.path.exists(fna_file), '{} does not exists'.format(fna_file)
        # self.labels = set()
        self.transform = transform
        self.match_dict = dict() # {hash_label1: [gene1, gene2,...], hash_label2: [gene10, gene11,...], ...,hash_labeln:...}
        self._len = 0
        with open(fna_file, 'r') as g_file:
            lines = g_file.readlines()
            lines = [line.strip() for line in lines]
            gene_str = ''
            hash_label = ''
            for line in lines:
                # Catch new sequence
                if line[0] == '>':

                    # Update hash label key with gene sting value
                    if hash_label != '':
                        self.match_dict[hash_label].append(ensure_gene_length(k_mer, gene_str))

                        # Track the number of genes
                        self._len += 1

                    # Reset hash_label for reading new sequence
                    hash_label = ''
                    gene_str = ''
                    dot_pos = line.find('.')
                    # Seq_flag indicate 1st or 2nd sequence
                    seq_flag = int(line[dot_pos + 1])

                    # 1st sequence, read the hash value (indicate the label)
                    if seq_flag == 1:
                        hash_pattern = re.search(GenomeDataset.HASH_PATTERN, line)
                        if hash_pattern is not None:
                            # for res in hash_pattern:
                            hash_label = hash_pattern.group(0)

                            # Remove the brackets
                            hash_label = hash_label.replace('(', '')
                            hash_label = hash_label.replace(')', '')
                            # self.labels.add(hash_label)
                            # break

                            # Create new mapping if new label is detected
                            if len(self.match_dict.get(hash_label, [])) == 0:
                                self.match_dict.update({hash_label: []})
                    else:
                        pass # Ignore 2nd sequence for now
                # Gene string
                else:
                    gene_str = gene_str + line

        # Process gene string to feature types
        if feature_type == 'bow':
            self.vocab = generate_k_mer_corpus(k_mer)
            data = []
            lb_length = []
            for key, genes in self.match_dict.items():
                temp = []
                for gene in genes:
                    # Insert space into every k_mer substring to make it looks like word
                    # or not insert anything if use overlapping kmer
                    separate_char = '' if overlap_k_mer else ' '
                    gene = separate_char.join(gene[i: i + k_mer] for i in range(0, len(gene), k_mer))

                    processed_gene = self.compute_bow_overlap(gene, k_mer) if overlap_k_mer else self.compute_bow(gene)
                    temp.append(processed_gene)

                data.extend(temp)
                lb_length.append(len(genes))

            prev = 0
            for i, item in enumerate(self.match_dict.items()):
                key, genes = item
                genes = data[prev: prev + lb_length[i]]
                prev += lb_length[i]

                if is_normalize:
                    genes = float(genes) * np.sqrt(len(genes[0]))
                    genes = normalize(genes, norm='l2') * 200.0
                self.match_dict[key] = genes

        # Preprocess labels
        label_list = list(self.match_dict.keys())
        self.lb_lookup = self.to_onehot_mapping_2(label_list)

    def compute_bow(self, gene_str: str):
        k_mer_parts = gene_str.split(' ')
        encode_vector = np.zeros(len(self.vocab))
        for part in k_mer_parts:
            pos = self.vocab.index(part)
            encode_vector[pos] += 1

        return encode_vector

    def compute_bow_overlap(self, gene_str: str, k_mer):
        encode_vector = np.zeros(len(self.vocab))
        for i in range(len(gene_str) - 4):
            sub_k_mer_str = gene_str[i: i + 4]
            try:
                idx = self.vocab.index(sub_k_mer_str)
            except ValueError:
                sub_k_mer_str = sub_k_mer_str + '_' * (k_mer - len(sub_k_mer_str))
                print(sub_k_mer_str)
                idx = self.vocab.index(sub_k_mer_str)
            encode_vector[idx] += 1

        return encode_vector
    
    def to_onehot_mapping(self, lb_list):
        length = len(lb_list)
        lb_mapping = dict()
        for i, lb in enumerate(lb_list):
            lb_mapping[lb] = np.zeros(length)
            lb_mapping[lb][i] = 1

        return lb_mapping

    def to_onehot_mapping_2(self, lb_list):
        lb_mapping = dict()
        for i, lb in enumerate(lb_list):
            lb_mapping[lb] = i

        return lb_mapping

    def __len__(self):
        # Return len of dataset in number of gene strings
        return self._len

    def __getitem__(self, idx):
        num_of_keys = len(self.match_dict.keys())
        previous_len = 0
        for key, genes in self.match_dict.items():
            genes_length = len(genes)
            if idx < genes_length + previous_len:
                data = genes[idx - previous_len]
                if self.transform:
                    data = self.transform(data)
                return (data, self.lb_lookup[key])
            else:
                previous_len += genes_length
                continue
        
        return (None, None)


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from transform.gene_transforms import numerize_genome_str
    metagene_dataset = GenomeDataset('data/gene/L1.fna', overlap_k_mer=False)
    for i in range(5):
        print(metagene_dataset.__getitem__(i))

    print('Total labels: ', metagene_dataset.match_dict.keys())