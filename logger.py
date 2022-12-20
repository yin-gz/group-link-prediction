import torch
import wandb

class Logger(object):
    """
    Each metric has a Logger, save the results in each valid time.

    """
    def __init__(self):
        self.results = []

    def add_result(self, result):
        """
        Append current epoch result to the Logger.

        Args:
            result: (valid_value, test_value)
        """
        #result
        self.results.append(result)