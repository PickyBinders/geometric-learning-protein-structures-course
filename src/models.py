from torch import nn
from torch_geometric import nn as graph_nn
import torch
import lightning

class GATModule(lightning.LightningModule):
    """
    LightningModule wrapping a GAT model.
    """
    def __init__(self):
        super().__init__()
        self.model = graph_nn.GAT(in_channels=20,
                         hidden_channels=32,
                         num_layers=2,
                         heads=2,
                         out_channels=1,
                         dropout=0.01,
                         jk="last", v2=True)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, node_attributes, edge_index):
        return self.model(node_attributes, edge_index)

    def training_step(self, batch, batch_idx):
        out = self(batch.amino_acid_one_hot.float(), batch.edge_index)
        loss = self.loss_function(out, batch.y.view(-1, 1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.batch_size)
        self.train_step_outputs.append((out.detach().cpu(), batch.y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.amino_acid_one_hot.float(), batch.edge_index)
        loss = self.loss_function(out, batch.y.view(-1, 1))
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.batch_size)
        self.validation_step_outputs.append((out.detach().cpu(), batch.y))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=0.001, weight_decay=0.0001)