import itertools as it
import numpy as np
import networkx as nx
import nxmetis
import copy

import sys
sys.path.append('.')

from dataloader.utils import load_meta_reads

LENGTH_OF_Q_MERS = 20   # q (short: 20, long: 30, 3species: 10)
NUM_SHARED_READS = 45    # m (short: 5, long: 45, 3species: 3)

# parameters for graph partitioning
MAXIMUM_COMPONENT_SIZE = 200 # R*, S*: 200

def build_overlap_graph(reads, labels, qmer_length=LENGTH_OF_Q_MERS, num_shared_reads=NUM_SHARED_READS):
    '''
    Build overlapping graph
    '''
    # Create hash table with q-mers are keys
    print("Building hash table...")
    lmers_dict=dict()
    for idx, r in enumerate(reads):
        for j in range(0,len(r)-qmer_length+1):
            lmer = r[j:j+qmer_length]
            if lmer in lmers_dict:
                lmers_dict[lmer] += [idx]
            else:
                lmers_dict[lmer] = [idx]

    print('Building edges...')
    E=dict()
    for lmer in lmers_dict:
        for e in it.combinations(lmers_dict[lmer],2):
            if e[0]!=e[1]:
                e_curr=(e[0],e[1])
            else:
                continue
            if e_curr in E:
                E[e_curr] += 1 # Number of connected lines between read a and b
            else:
                E[e_curr] = 1
    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
    
    print('Building graph...')
    G = nx.Graph()
    print('Adding nodes...')
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'darkcyan', 5: 'violet'}
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i], color=color_map[labels[i]])

    print('Adding edges...')
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])
    print('Graph constructed!')
    
    return G

def metis_partition_groups_seeds(G, only_seed=False, maximum_group_size=MAXIMUM_COMPONENT_SIZE):
    CC = [cc for cc in nx.connected_components(G)]
    GL = []
    for subV in CC:
        if len(subV) > maximum_group_size:
            # use metis to split the graph
            subG = nx.subgraph(G, subV)
            nparts = int(len(subV)/maximum_group_size + 1)
            (edgecuts, parts) = nxmetis.partition(subG, nparts)
            # add to group list
            GL += parts
        else:
            GL += [list(subV)]

    SL = []
    if only_seed:
        for p in GL:
            pG = nx.subgraph(G, p)
            SL += [nx.maximal_independent_set(G)]

    return GL, SL

def bimeta_partition(G: nx.Graph, maximum_seeds=MAXIMUM_COMPONENT_SIZE):
    GL = []
    SL = []
    temp_G = copy.copy(G)
    traversed_nodes = []
    for i, node in enumerate(temp_G.nodes):
        if node in traversed_nodes:
            continue
        SGi = []
        Gi = []
        SGi.append(node)
        Gi.append(node)
        traversed_nodes.append(node)
        for grp_node in Gi:
            neighbors = list(temp_G.neighbors(grp_node))
            if neighbors:
                add_node = None
                for n_node in neighbors:
                    if n_node not in traversed_nodes:
                        add_node = n_node
                        traversed_nodes.append(n_node)
                        break

                # random_first_neighbor = neighbors[0]
                # # temp_G.remove_node(random_first_neighbor)
                # traversed_nodes.append(random_first_neighbor)
                if add_node:
                    if add_node not in SGi:
                        SGi.append(add_node)
                    Gi.append(add_node)

            if (len(SGi) >= maximum_seeds) or (not temp_G.nodes):
                break

        SL.append(SGi)
        GL.append(Gi)

    return GL, SL

def show_graph(graph, num_of_species):
    plt.figure(figsize=(10,10))
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'tomato', 5: 'violet'}
    print("#nodes = %d" % (len(graph.nodes())))
    print("#edges = %d" % (len(graph.edges())))
    pos = nx.spring_layout(graph)  # positions for all nodes
    for l in range(0, num_of_species):
        nodelist = [n for n in graph.nodes() if labels[n] == l]
        nx.draw_networkx_nodes(graph, pos,
                                nodelist=nodelist,
                                node_shape='.',
                                node_color=color_map[l],
                                node_size=20)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    import sys, json
    import matplotlib.pyplot as plt
    from networkx.readwrite import json_graph
    sys.path.append('.')
    data = 'data/gene/S2.fna'
    NUM_OF_SPECIES = 2
    reads, labels = load_meta_reads(data, type='fasta')

    # print(len([label for label in labels if label == 1]))
    # print(len([labels == 0]))

    G1 = build_overlap_graph(reads, labels, 20)
    # G2 = build_overlap_graph(reads[:10], labels[:10])
    # G3 = build_overlap_graph(reads[:10], labels[:10])
    # G4 = build_overlap_graph(reads[:10], labels[:10])
    # G5 = build_overlap_graph(reads[:10], labels[:10])
    

    Gs = [G1]
    g_data_s = []
    # for i, G in enumerate(Gs):
    #     g_data = json_graph.node_link_data(G)
    #     g_data_s.append(g_data)
    g_data = json_graph.node_link_data(G1)

    # for i, g_data in enumerate(g_data_s):
    with open('S2_graph.json', 'w') as f:
        json.dump(g_data, f)
            

    # with open('gs.json', 'r') as f:
        # json.dump(g_data, f)
        # gdata_s = json.load(f)

    # graphs = []
    # for g_data in g_data_s:
    #     graph = json_graph.node_link_graph(g_data)
    #     show_graph(graph, NUM_OF_SPECIES)