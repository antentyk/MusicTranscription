import torch
import torch.utils.data

import tqdm

from src import config, get_logger, round_probabilities, Dataset, get_metrics

logger = get_logger()

logger.info("Loading train dataset")
trainDataset = Dataset(config["path_to_processed_MAPS"], "train")
trainDatasetMean = trainDataset.mean()

logger.info("Loading test dataset")
testDataset = Dataset(config["path_to_processed_MAPS"], "test")

dataloader = torch.utils.data.DataLoader(testDataset, batch_size=config["mini_batch_size"], shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("./models/Dnn3Layers/095.pth")
model = model.to(device)
model = model.eval()

testMetrics = {}

for batchX, batchy in tqdm.tqdm(dataloader):
	batchX -= trainDatasetMean
	batchX = batchX.to(device)
	batchy = batchy.to(device)

	prediction = round_probabilities(model(batchX))

	batchMetrics = get_metrics(prediction, batchy)

	for metric in config["metrics_names"]:
		testMetrics[metric] = testMetrics.get(metric, 0) + batchMetrics[metric]

for metric in testMetrics:
	logger.info("%s: %s" % (metric, testMetrics[metric] / len(testDataset)))

