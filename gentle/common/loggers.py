import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Logger(object):
    """  Logs data and pushes to TensorBoard  """

    def __init__(self, output_directory, device='cpu'):
        """  Construct LoggerMPI object  """
        self.summary_writer = SummaryWriter(log_dir=output_directory)
        self.graph_logged = False
        self.device = device

    def log_scalar(self, key, value, x):
        """  Logs a scalar y value, using MPI to determine x value  """
        self.summary_writer.add_scalar(key, value, x)

    def log_mean_value(self, key, value, x):
        """  Adds the mean of a given data list to the log  """
        if len(value) > 0:
            self.summary_writer.add_scalar(key, np.mean(value), x)

    def log_config(self, config_obj):
        config_str = json.dumps(config_obj, indent=2, cls=NumpyEncoder)
        config_str = "".join("\t" + line for line in config_str.splitlines(True))
        self.summary_writer.add_text('config', config_str, global_step=0)

    def log_graph(self, observations, network):
        """  Initialize TensorBoard logging of model graph """
        if not self.graph_logged:
            input_obs = torch.from_numpy(observations).float().to(self.device)
            self.summary_writer.add_graph(network, input_obs)
        self.graph_logged = True

    def flush(self):
        self.summary_writer.flush()


class JSONLogger(object):
    ''' 
    logs data to a json file in case we want to examine or plot it after-the-fact
    JSON file has structure:
    "key":{"x":list_of_timesteps, "values":list_of_values}
    '''

    def __init__(self, output_directory):
        self.data = {}
        self.output_directory = output_directory

    def log_scalar(self, key, value, x):
        if key not in self.data:
            self.data[key] = {"x":[], "values":[]}
        self.data[key]["x"].append(x)
        self.data[key]["values"].append(value)

    def log_mean_value(self, key, values, x):
        if isinstance(values, dict):
            return

        if key not in self.data:
            self.data[key] = {"x":[], "values":[]}
        self.data[key]["x"].append(x)
        self.data[key]["values"].append(np.mean(values))

    def flush(self):
        # save json file
        with open(self.output_directory+'/logs.json', 'w') as f:
            json.dump(self.data, f)