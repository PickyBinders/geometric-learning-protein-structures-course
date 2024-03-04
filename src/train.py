import dataloader, models
from lightning.pytorch.cli import LightningCLI
from loguru import logger


def main():
    """
    Run with python train.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    LightningCLI(models.GATModule, dataloader.ProteinGraphDataModule)

if __name__ == "__main__":
    main()