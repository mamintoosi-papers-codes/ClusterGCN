import torch
import numpy as np
import pandas as pd
import networkx as nx
from texttable import Texttable
from scipy.sparse import coo_matrix
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikiCS
from torch_geometric.utils.convert import to_networkx

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph

def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index)+1
    feature_count = max(feature_index)+1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    return features
    # Number of graph nodes:  19717
    # (19717, 500) (19717, 1)

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"]).reshape(-1,1)
    return target

def dataset_reader(args):
    """
    Reading the dataset
    :param dataset_name: Name of the dataset.
    :param path: Path to the target.
    :return target: Target vector.
    """
    dataset_name = args.dataset_name
    # if dataset_name=='PPI':        
    #     dataset = PPI(root=path)
    if dataset_name=='default':
        graph = graph_reader("./input/edges.csv")
        features = feature_reader("./input/features.csv")
        target = target_reader("./input/target.csv")

    elif dataset_name in ['PubMed', 'Cora', 'CiteSeer','WikiCS']:
        if dataset_name == 'PubMed':
            dataset = Planetoid(root=args.ds_root+'/PubMed', name='PubMed', split='full')
        elif dataset_name == 'Cora':
            dataset = Planetoid(root=args.ds_root+'/Cora', name='Cora', split='full')
        elif dataset_name == 'CiteSeer':
            dataset = Planetoid(root=args.ds_root+'/CiteSeer', name='CiteSeer', split='full')
        elif dataset_name == 'WikiCS':
            dataset = WikiCS(root=args.ds_root+'/WikiCS')
        data = dataset[0]
        graph = to_networkx(data, to_undirected=True)
        # node_labels = data.y[list(graph.nodes)].numpy()
        # len(graph.nodes()), len(graph.edges())
        features = data.x.numpy()
        target = data.y.numpy()[..., np.newaxis]

    return graph, features, target
