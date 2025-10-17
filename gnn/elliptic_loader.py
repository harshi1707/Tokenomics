import os
import pandas as pd
import networkx as nx


def load_elliptic(csv_dir: str, graph_file: str, features_file: str, classes_file: str):
    edges_path = os.path.join(csv_dir, graph_file)
    features_path = os.path.join(csv_dir, features_file)
    classes_path = os.path.join(csv_dir, classes_file)

    edges = pd.read_csv(edges_path)
    features = pd.read_csv(features_path)
    classes = pd.read_csv(classes_path)

    G = nx.from_pandas_edgelist(edges, source='txId1', target='txId2')
    return G, features, classes

