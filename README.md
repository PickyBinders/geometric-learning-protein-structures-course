# Geometric Deep Learning for Protein Structure Data with PyTorch Lightning

- `10:00 - 10:15`: [Introduction](0_Objectives.ipynb)
- `10:15 - 11:00`: [Notebook 1 - Proteins as Graphs](1_Proteins_as_Graphs.ipynb)
- `11:00 - 11:30`: _Break_
- `11:30 - 12:30`: [Notebook 2 - Graph Datasets and DataLoaders](2_Graph_Datasets.ipynb)
- `12:30 - 13:30`: _Lunch_
- `13:30 - 13:45`: Introduction to geometric deep learning
- `13:45 - 15:00`: [Notebook 3 - Geometric Deep Learning](3_Geometric_Deep_Learning_Models.ipynb)
- `15:00 - 15:30`: _Break_
- `15:30 - 16:30`: [Notebook 4 - Training and Tracking](4_Training_and_Tracking.ipynb)
- `16:30 - 17:00`: Wrap-up

Link to the google colab search for github repos: https://colab.research.google.com/github/ 
with the repository: https://github.com/PickyBinders/geometric-learning-protein-structures-course

## Objectives
Develop a code-base for exploring, training and evaluating graph deep learning models using protein structures as input for a residue-level prediction task.
- Learn how to featurize protein structures as graphs using [Graphein](https://graphein.ai/)
- Understand the data loading and processing pipeline for graph datasets using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- Learn how to implement graph neural networks using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- Understand the typical deep learning training and evaluation loops using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## Task and Dataset

- **Given an input protein chain, predict for each residue whether or not it belongs to a protein-protein interface.**
- The dataset (in `dataset.txt`) is a subset of the [MaSIF-site dataset](https://www.nature.com/articles/s41592-019-0666-6). 
- Each line is a PDB ID and a chain. We'll use these to extract residues at the interface with other chains and label them as positive examples. All other residues are negative examples.


