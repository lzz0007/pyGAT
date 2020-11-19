from gnn_cnn_model.graph import *
from gnn_cnn_model.utils import *


def deepwalks(edges, undirected, number_walks, walk_length, seed):

    # load edgelist into the Graph
    G = Graph()
    for i in range(edges.shape[0]):
        h, t = edges[i, 0], edges[i, 1]
        G[int(h)].append(int(t))
        if undirected:
            G[int(t)].append(int(h))

    G.make_consistent()

    # print("Number of nodes: {}".format(len(G.nodes())))
    num_walks = len(G.nodes()) * number_walks
    # print("Number of walks: {}".format(num_walks))
    data_size = num_walks * walk_length
    # print("Data size (walks*length): {}".format(data_size))

    # print("Walking...")
    # for each node, sample n paths
    walks = build_deepwalk_corpus(G, num_paths=number_walks,
                                  path_length=walk_length, alpha=0, rand=random.Random(seed))
    return walks

