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

        # mrr result
        self.best_val_mrr = [0. for _ in range(runs)]#best_mrr_epoch
        self.best_val_mrr_results = [0. for _ in range(runs)]
        self.best_test_mrr_results = [0. for _ in range(runs)]

    def add_result(self, run_id, result):
        #result (valid_value, test_value)
        self.results[run_id].append(result)

    def print_statistics(self, metrics, run_id=None, use_wandb = False, best_mrr_epoch = 0):
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

            #mrr result
            self.best_val_mrr[run_id] = best_mrr_epoch
            self.best_val_mrr_results[run_id] = round(result[best_mrr_epoch, 0].item(), 2)
            self.best_test_mrr_results[run_id] = round(result[best_mrr_epoch, 1].item(), 2)

            if use_wandb:
                wandb.run.summary[f"Run＿{run_id}:　best/Valid/{metrics}"] = best_valid
                wandb.run.summary[f"Run＿{run_id}:　best/Test/{metrics}"] = best_test
                wandb.run.summary[f"Run＿{run_id}:　best_{metrics}_epoch"] = argmax
                wandb.run.summary[f"Run＿{run_id}:　best_mrr/Valid/{metrics}"] = self.best_val_mrr_results[run_id]
                wandb.run.summary[f"Run＿{run_id}:　best_mrr/Test/{metrics}"] = self.best_test_mrr_results[run_id]
        else:
            val_results = np.array(self.best_val_results)
            test_results = np.array(self.best_test_results)

            print(f'All Runs Results:')
            print(f'Highest Valid: {val_results.mean():.2f} ± {val_results.std():.2f}')
            print(f'Final Test: {test_results.mean():.2f} ± {test_results.std():.2f}')
            print(f'MRR All Runs Results:')

            #mrr
            mrr_val_results = np.array(self.best_val_mrr_results)
            mrr_test_results = np.array(self.best_test_mrr_results)
            print(f'All Runs Results(MRR):')
            print(f'Highest Valid: {mrr_val_results.mean():.2f} ± {mrr_val_results.std():.2f}')
            print(f'Final Test: {mrr_test_results.mean():.2f} ± {mrr_test_results.std():.2f}')


            if use_wandb:
                wandb.run.summary[f"total_best/Valid/{metrics}"] = str(round(val_results.mean(), 2)) + '±' + str(round(val_results.std(), 2))
                wandb.run.summary[f"total_best/Test/{metrics}"] = str(round(test_results.mean(), 2)) + '±' + str(round(test_results.std(), 2))
                wandb.run.summary[f"total_best_mrr/Valid/{metrics}"] = str(round(mrr_val_results.mean(), 2)) + '±' + str(round(mrr_val_results.std(), 2))
                wandb.run.summary[f"total_best_mrr/Test/{metrics}"] = str(round(mrr_test_results.mean(), 2)) + '±' + str(round(mrr_test_results.std(), 2))
