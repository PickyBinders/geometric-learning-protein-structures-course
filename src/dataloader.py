import graphein
graphein.verbose(enabled=False)
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes import amino_acid as graphein_nodes
from graphein.protein import edges as graphein_edges
from graphein.protein.subgraphs import extract_subgraph
from graphein.ml import GraphFormatConvertor
from functools import partial
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import pickle
import torch
import lightning as L
from torch.utils.data import random_split
from Bio.PDB.mmtf import MMTFParser
from Bio.PDB.SASA import ShrakeRupley


def load_graph(pdb_id, chain):
    graph_config = ProteinGraphConfig(
        node_metadata_functions = [graphein_nodes.amino_acid_one_hot, graphein_nodes.meiler_embedding],
        edge_construction_functions = [graphein_edges.add_peptide_bonds, 
                                       partial(graphein_edges.add_distance_threshold, threshold=8., long_interaction_threshold=2)],
    )
    graph = construct_graph(pdb_code=pdb_id, config=graph_config, verbose=False)

    struct = MMTFParser.get_structure_from_url(pdb_id)
    sr = ShrakeRupley()
    sr.compute(struct, level="R") #residue level
     
    for node, data in graph.nodes(data=True):
        chain_id = data['chain_id']
        residue_number = data['residue_number']
        data['sasa'] = round(struct[0][chain_id][residue_number].sasa, 2)

    interface_residues = set()
    for source, target, kind in graph.edges(data=True):
        c1, c2 = source.split(":")[0], target.split(":")[0]
        if 'distance_threshold' in kind['kind']:
            if c1 == chain and c2 != chain:
                interface_residues.add(source)
            elif c2 == chain and c1 != chain:
                interface_residues.add(target)
    return extract_subgraph(graph, chains=chain), interface_residues


def graphein_to_torch_graph(graphein_graph, interface_labels, convertor, 
                              node_attr_columns = ["amino_acid_one_hot", "meiler"]):
    """
    Converts a Graphein graph to a pytorch-geometric Data object.
    """
    data = convertor(graphein_graph)
    data_dict= data.to_dict()
    x_data = []
    for x in node_attr_columns:
        if data_dict[x].ndim == 1:  #if we want to include solvent accesible area, which is one-dimensional
            x_data.append(torch.atleast_2d(data_dict[x]).T)
        else:
            x_data.append(torch.atleast_2d(data_dict[x]))
    data.x = torch.hstack(x_data).float()
    data.pos = data.coords.float()
    data.y = torch.zeros(data.num_nodes)
    for i, node_id in enumerate(data.node_id):
        if node_id in interface_labels:
            data.y[i] = 1
    return data

class ProteinDataset(Dataset):
    """
    torch-geometric Dataset class for loading protein files as graphs.
    """
    def __init__(self, root,
                 protein_names: list,
                 columns: list):

        self.convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=columns, verbose = None)
        self.protein_names = protein_names
        super(ProteinDataset, self).__init__(root)

    def download(self):
        for protein_name in self.protein_names:
            output = Path(self.raw_dir) / f'{protein_name}.pkl'
            if not output.exists():
                pdb_id, chain = protein_name.split("_")
                graphein_graph, interface_labels = load_graph(pdb_id, chain)
                with open(output, "wb") as f:
                    pickle.dump((graphein_graph, interface_labels), f)

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
                    graphein_graph, interface_labels = pickle.load(f)
                torch_graph = graphein_to_torch_graph(graphein_graph, interface_labels, convertor=self.convertor)
                torch.save(torch_graph, output)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data

class ProteinGraphDataModule(L.LightningDataModule):
    def __init__(self, root, columns, dataset_file, batch_size=8, num_workers=4):
        super().__init__()
        self.root = root
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        with open(dataset_file) as f:
            self.protein_names = [line.strip() for line in f]
        self.protein_names = self.protein_names[:100]
        self.columns = columns

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed settings
        # does the downloading and saving of graphein graphs part, just once
        ProteinDataset(root=self.root,
                        protein_names=self.protein_names,
                        columns=self.columns)
    

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # now it's just loaded and not downloaded processed etc.
        dataset = ProteinDataset(root=self.root,
                                  protein_names=self.protein_names,
                                  columns=self.columns)
        train_idx, val_idx, test_idx = random_split(range(len(dataset)), [0.8, 0.1, 0.1])
        self.train, self.val, self.test = dataset[list(train_idx)], dataset[list(val_idx)], dataset[list(test_idx)]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)