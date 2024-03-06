import graphein
graphein.verbose(enabled=False)
import warnings
warnings.filterwarnings("ignore")
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes import amino_acid as graphein_nodes
from graphein.protein import edges as graphein_edges
from graphein.protein.subgraphs import extract_subgraph
from functools import partial
from graphein.ml import GraphFormatConvertor
import torch
import lightning
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from pathlib import Path
import pickle

def load_graph(pdb_id, chain):
    graph_config = ProteinGraphConfig(
        node_metadata_functions = [graphein_nodes.amino_acid_one_hot, graphein_nodes.meiler_embedding],
        edge_construction_functions = [graphein_edges.add_peptide_bonds, 
                                       partial(graphein_edges.add_distance_threshold, 
                                               threshold=8.,
                                               long_interaction_threshold=2)],
    )
    graph = construct_graph(pdb_code=pdb_id, config=graph_config, verbose=False)
    interface_residues = set()
    for source, target, kind in graph.edges(data=True):
        c1, c2 = source.split(":")[0], target.split(":")[0]
        if 'distance_threshold' in kind['kind']:
            if c1 == chain and c2 != chain:
                interface_residues.add(source)
            elif c2 == chain and c1 != chain:
                interface_residues.add(target)
    graph = extract_subgraph(graph, chains=chain)
    for node, data in graph.nodes(data=True):
        if node in interface_residues:
            data['interface_label'] = 1
        else:
            data['interface_label'] = 0
    return graph

class ProteinDataset(Dataset):
    """
    torch-geometric Dataset class for loading protein files as graphs.
    """
    def __init__(self, root,
                 protein_names: list):
        columns = [
            "chain_id",
            "coords",
            "edge_index",
            "kind",
            "node_id",
            "residue_number",
            "amino_acid_one_hot",
            "meiler",
            "interface_label",
        ]
        self.convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=columns, verbose=None)
        self.protein_names = protein_names
        super(ProteinDataset, self).__init__(root)

    def download(self):
        for protein_name in self.protein_names:
            output = Path(self.raw_dir) / f'{protein_name}.pkl'
            if not output.exists():
                pdb_id, chain = protein_name.split("_")
                graphein_graph = load_graph(pdb_id, chain)
                with open(output, "wb") as f:
                    pickle.dump(graphein_graph, f)

    @property
    def raw_file_names(self):
        return [Path(self.raw_dir) / f"{protein_name}.pkl" for protein_name in self.protein_names if (Path(self.raw_dir) / f"{protein_name}.pt").exists()]

    @property
    def processed_file_names(self):
        return [Path(self.processed_dir) / f"{protein_name}.pt" for protein_name in self.protein_names if (Path(self.processed_dir) / f"{protein_name}.pt").exists()]

    def process(self):
        for protein_name in self.protein_names:
            output = Path(self.processed_dir) / f'{protein_name}.pt'
            if not output.exists():
                with open(Path(self.raw_dir) / f"{protein_name}.pkl", "rb") as f:
                    graphein_graph = pickle.load(f)
                torch_data = self.convertor(graphein_graph)
                torch.save(torch_data, output)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data


class ProteinGraphDataModule(lightning.LightningDataModule):
    def __init__(self, root, dataset_file, batch_size=8):
        super().__init__()
        self.root = root
        self.dataset_file = dataset_file
        with open(dataset_file) as f:
            self.protein_names = [line.strip() for line in f]
        self.protein_names = self.protein_names[:20] # SMALL DATASET FOR TESTING
        self.batch_size = batch_size

    def prepare_data(self):
        ProteinDataset(root=self.root, protein_names=self.protein_names)
    
    def setup(self, stage):
        dataset = ProteinDataset(root=self.root, protein_names=self.protein_names)
        # Here we just do a random split of 80/10/10 for train/val/test
        train_idx, val_idx, test_idx = random_split(range(len(dataset)), [0.8, 0.1, 0.1])
        self.train, self.val, self.test = dataset[list(train_idx)], dataset[list(val_idx)], dataset[list(test_idx)]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)