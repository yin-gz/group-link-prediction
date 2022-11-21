import torch
import wandb
import numpy as np

# each metric has a Logger, save the results in each valid time
#after each run, print the test results according to the best valid results
class Logger(object):
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]
        self.best_val_results = [0. for _ in range(runs)]
        self.best_test_results = [0. for _ in range(runs)]

    def add_result(self, run_id, result):
        #result (valid_value, test_value)
        self.results[run_id].append(result)

    def print_statistics(self, metrics, run_id=None, use_wandb = False):
        if run_id is not None:
            result = torch.tensor(self.results[run_id]) # [epoch, 2]
            argmax = result[:, 0].argmax().item()
            best_valid = round(result[argmax, 0].item(), 2)
            best_test = round(result[argmax, 1].item(), 2)
            print(f'Run {run_id + 1:02d}:')
            print(f'Highest Valid: {best_valid}')
            print(f'Final Test: {best_test}')
            self.best_val_results[run_id] = best_valid
            self.best_test_results[run_id] = best_test
            if use_wandb:
                wandb.run.summary[f"Run＿{run_id}:　best/Valid/{metrics}"] = best_valid
                wandb.run.summary[f"Run＿{run_id}:　best/Test/{metrics}"] = best_test
                wandb.run.summary[f"Run＿{run_id}:　best/{metrics}_epoch"] = argmax

        else:
            val_results = np.array(self.best_val_results)
            test_results = np.array(self.best_test_results)

            print(f'All runs:')
            print(f'Highest Valid: {val_results.mean():.2f} ± {val_results.std():.2f}')
            print(f'Final Test: {test_results.mean():.2f} ± {test_results.std():.2f}')

            if use_wandb:
                wandb.run.summary[f"best/Valid/{metrics}"] = val_results.mean()
                wandb.run.summary[f"best/Test/{metrics}"] = test_results.mean()
