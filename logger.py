import torch
import wandb

class Logger(object):
    """
    Each metric has a Logger, save the results in each valid time.

    """
    def __init__(self):
        self.results = []
        self.best_val_results = 0.
        self.best_test_results = 0.

    def add_result(self, result):
        """
        Append current epoch result to the Logger.

        Args:
            result: (valid_value, test_value)
        """
        #result
        self.results.append(result)

    def print_statistics(self, metrics, use_wandb = False):
        """
        After each run, print the test results according to the best valid results.

        """
        result = torch.tensor(self.results) # [epoch, 2]
        argmax = result[:, 0].argmax().item()
        best_valid = round(result[argmax, 0].item(), 2)
        best_test = round(result[argmax, 1].item(), 2)
        print(f'Highest Valid: {best_valid}')
        print(f'Final Test: {best_test}')
        self.best_val_results = best_valid
        self.best_test_results = best_test

        if use_wandb:
            wandb.run.summary[f"Best/Valid/{metrics}"] = best_valid
            wandb.run.summary[f"Best/Test/{metrics}"] = best_test
            wandb.run.summary[f"Best/Test_epoch/{metrics}"] = argmax
