import json

from utils.evaluation import batch_means_ess
import numpy as np

path = "path/to/samples"
results_path = "results/experiment_name"

results = np.load(path)
samples = results['arr_0']
train_time = results['arr_1']
sample_time = results['arr_2']

ess = batch_means_ess(samples)

result_dict = {'ess:': ess,
               'ess/sample_time:': ess/sample_time,
               'ess/total_time:': ess/(sample_time + total_time),
               'sample_time:': sample_time,
               'train_time': train_time}

with open(results_path) as f:
    json.dump(result_dict, f)