import numpy as np
from itertools import combinations, combinations_with_replacement, permutations

NUCLEOTIDS = ['A', 'T', 'G', 'C']
def generate_k_mer_corpus(k_val, base_corpus: list=NUCLEOTIDS):
    def sort_corpus(x: str):
        '''
        Make corpus ordered in the number of _
        '''
        nums = x.count('_')
        return nums

    all_length_k_val = np.array([len(base_corpus) ** (i + 1) for i in range(k_val)])
    corpus_length = np.sum(all_length_k_val).tolist()
    # corpus_length = len(base_corpus) ** k_val
    corpus = []
    shuffled_base_corpuses = combinations_with_replacement(base_corpus, k_val)

    for shuffled_corpus in shuffled_base_corpuses:
        for i in range(k_val):
            combs = combinations_with_replacement(shuffled_corpus, i + 1)
            perms = permutations(shuffled_corpus, i + 1)
            for elem in combs:
                corpus.append(ensure_gene_length(k_val, ''.join(elem)))
            for elem in perms:
                corpus.append(ensure_gene_length(k_val, ''.join(elem)))

    corpus = list(set(corpus))
    corpus = sorted(corpus, key=sort_corpus)

    assert corpus_length == len(corpus), 'Generated corpus failed. Inconsistent of generated\
         length with expected length: {} and {}'.format(len(corpus), corpus_length)

    return corpus

def ensure_gene_length(k_mer_val, gene_str):
    unfull_length =  len(gene_str) % k_mer_val
    gene_str = gene_str + '_' * (k_mer_val - unfull_length) if unfull_length != 0 else gene_str

    return gene_str

def test_ensure_gene_length():
    gene_str1 = 'ATGCTACGTACTAG'
    gene_str2 = 'ATGCTACGTACTA'
    gene_str3 = 'ATGCTACGTACT'
    k_mer = 4
    print(ensure_gene_length(k_mer, gene_str1))
    print(ensure_gene_length(k_mer, gene_str2))
    print(ensure_gene_length(k_mer, gene_str3))

def test_generate_corpus():
    corpus = generate_k_mer_corpus(4, NUCLEOTIDS)
    print('=================== Corpus content ====================')
    print(corpus)
    print('=================== Consistent length of corpus ====================')
    print('Expected length: ', len(corpus))

if __name__ == "__main__":
    test_generate_corpus()
    # test_ensure_gene_length()

