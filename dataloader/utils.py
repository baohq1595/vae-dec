import numpy as np
from itertools import combinations, combinations_with_replacement, permutations
from Bio import SeqIO
from Bio.Seq import Seq
import re
import gensim
from gensim import corpora
import itertools as it
import json

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

def load_meta_reads(filename, type='fasta'):
    def format_read(read):
        # Return sequence and label
        z = re.split('[|={,]+', read.description)
        return read.seq, z[3]
    try:
        seqs = list(SeqIO.parse(filename, type))
        reads = []
        labels = []

        # Detect for paired-end or single-end reads
        # If the id of two first reads are different (e.g.: .1 and .2), they are paired-end reads
        is_paired_end = False
        if len(seqs) > 2 and seqs[0].id[-1:] != seqs[1].id[-1:]:
            is_paired_end = True

        label_list = dict()
        label_index = 0

        for i in range(0, len(seqs), 2 if is_paired_end else 1):
            read, label = format_read(seqs[i])
            if is_paired_end:
                read2, label2 = format_read(seqs[i + 1])
                read += read2
            reads += [str(read)]

            # Create labels
            if label not in label_list:
                label_list[label] = label_index
                label_index += 1
            labels.append(label_list[label])

        del seqs

        return reads, labels
    except Exception as e:
        print('Error when loading file {} '.format(filename))
        print('Cause: ', e)
        return []

def gen_kmers(klist):
    '''
    Generate list of k-mer words. Given multiple k-mer values.
    Args:
        klist: list of k-mer value
    Return:
        List of k-mer words
    '''
    bases = ['A', 'C', 'G', 'T']
    kmers_list = []
    for k in klist:
        kmers_list += [''.join(p) for p in it.product(bases, repeat=k)]

    # reduce a half of k-mers due to symmetry
    kmers_dict = dict()
    for myk in kmers_list:
        k_reverse_complement=Seq(myk).reverse_complement()
        if not myk in kmers_dict and not str(k_reverse_complement) in kmers_dict:
            kmers_dict[myk]=0

    return list(kmers_dict.keys())

def create_document( reads, klist):
    """
    Create a set of document from reads, consist of all k-mer in each read
    For example:
    k = [3, 4, 5]
    documents =
    [
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 1
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 2
        ...
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read n
    ]
    :param reads:
    :param klist: list of int
    :return: list of str
    """
    # create a set of document
    documents = []
    for read in reads:
        k_mers_read = []
        for k in klist:
            k_mers_read += [read[j:j + k] for j in range(0, len(read) - k + 1)]
        documents.append(k_mers_read)

    k_mers_set = [gen_kmers(klist)]
    dictionary = corpora.Dictionary(k_mers_set)
    return dictionary, documents

def save_documents(documents, file_path):
    with open(file_path, 'w') as f:
        for d in documents:
            f.write("%s\n" % d)


def parallel_create_document(reads, klist, n_workers=2 ):
    """
    Create a set of document from reads, consist of all k-mer in each read
    For example:
    k = [3, 4, 5]
    documents =
    [
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 1
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 2
        ...
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read n
    ]
    :param reads:
    :param klist: list of int
    :return: list of str
    """

    # create k-mer dictionary
    k_mers_set = [gen_kmers( klist )] #[genkmers(val) for val in klist]
    dictionary = corpora.Dictionary(k_mers_set)

    documents = []
    reads_str_chunk = [list(item) for item in np.array_split(reads, n_workers)]
    chunks = [(reads_str_chunk[i], klist) for i in range(n_workers)]
    pool = Pool(processes=n_workers)

    result = pool.starmap(create_document, chunks)
    for item in result:
        documents += item
    return dictionary, documents

def create_corpus(dictionary: corpora.Dictionary, documents, 
                  is_tfidf=False, 
                  smartirs=None, 
                  is_log_entropy=False, 
                  is_normalize=True):
    corpus = [dictionary.doc2bow(d, allow_update=False) for d in documents]
    if is_tfidf:
        tfidf = TfidfModel(corpus=corpus, smartirs=smartirs)
        corpus = tfidf[corpus]
    elif is_log_entropy:
        log_entropy_model = LogEntropyModel(corpus, normalize=is_normalize)
        corpus = log_entropy_model[corpus]

    return corpus

def compute_kmer_dist(dictionary, corpus, groups, seeds, only_seed=True):
    corpus_m = gensim.matutils.corpus2dense(corpus, len(dictionary.keys())).T
    res = []
    if only_seed:
        for seednodes in seeds:
            tmp = corpus_m[seednodes, :]
            res += [np.mean(tmp, axis=0)]
    else:
        for groupnodes in groups:
            tmp = corpus_m[groupnodes, :]
            res += [np.mean(tmp, axis=0)]
    return np.array(res)

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

def test_kmer_graph():
    # Test gen_kmers function, expected 136 words generated for 4-mer (already exclude complement kmer)
    kmwords = gen_kmers([4])
    # print(kmwords)
    # print('======================================')
    # print(len(kmwords))

    # Test create_document function
    data = 'data/gene/S2.fna'
    NUM_OF_SPECIES = 2
    reads, labels = load_meta_reads(data, type='fasta')
    dictionary, documents = create_document(reads, [4])

    # print('############# Dictionary ###############\n', dictionary)
    # print('############# Documents ################\n', documents[:10], '....')

    # Test create_corpus function
    corpus = create_corpus(dictionary, documents)
    # print('############# corpus ###############\n', corpus)

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    # test_generate_corpus()
    # test_ensure_gene_length()
    test_kmer_graph()

