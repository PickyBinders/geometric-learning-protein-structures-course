import dataloader, models
from lightning.pytorch.cli import LightningCLI


def main():
    LightningCLI(models.GATModule, dataloader.ProteinGraphDataModule)

if __name__ == "__main__":
    main()