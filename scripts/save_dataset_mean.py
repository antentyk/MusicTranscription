import torch
import os

from src import Dataset, config

trainDataset = Dataset(config["path_to_processed_MAPS"], "train")

print("Train dataset means:")
print(trainDataset.mean())

print("Saving means...")
torch.save(trainDataset.mean(), os.path.join(config["public_models_folder"], "train_means.pth"))

print("Done!")