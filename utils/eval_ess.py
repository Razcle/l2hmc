from utils.evaluation import batch_means_ess
import numpy as np

path = "path/to/samples"

samples = np.load(path)

ess = batch_means_ess(samples)

print(ess)