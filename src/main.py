import torch
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, dataset_reader
import numpy as np
# graph_reader, feature_reader, target_reader

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph, features, target = dataset_reader(args)
    # graph = graph_reader(args.edge_path)
    # # print('Number of graph nodes: ',len(graph.nodes()))
    # features = feature_reader(args.features_path)
    # target = target_reader(args.target_path)
    print(features.shape, target.shape)
    Scores = []
    for i in range(args.num_trial):
        clustering_machine = ClusteringMachine(args, graph, features, target)
        clustering_machine.decompose()
        gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
        gcn_trainer.train()
        score = gcn_trainer.test()
        Scores.append(score)
        print("\nF-1 score: {:.4f}".format(score))

    print("\n\n Mean F-1 score: {:.4f}".format(np.mean(Scores)))

if __name__ == "__main__":
    main()
