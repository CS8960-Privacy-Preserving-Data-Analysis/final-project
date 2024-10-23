import os

import torch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def log_metrics(args, best_prec1, train_time, save_dir, experiment_id, train_losses, val_losses, train_accuracies, val_accuracies, final_epsilon):
    """
    Logs the training metrics to a file.
    """
    # Save metrics to a log file
    log_file_path = os.path.join(save_dir, "metrics.log")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Experiment ID: {experiment_id}\n")
        log_file.write(f"Training epochs: {args.epochs}\n")
        log_file.write(f"Batch size: {args.batch_size}\n")
        log_file.write(f"Learning rate: {args.lr}\n")
        log_file.write(f"Target epsilon: {args.target_epsilon}\n")
        log_file.write(f"Final epsilon: {final_epsilon:.2f}\n")
        log_file.write(f"Total training time: {train_time:.2f} seconds\n")
        log_file.write(f"Best accuracy: {best_prec1:.2f}%\n")
        log_file.write(f"Model saved at: {os.path.abspath(args.save_dir)}\n")
        log_file.write(f"Training losses: {train_losses}\n")
        log_file.write(f"Validation losses: {val_losses}\n")
        log_file.write(f"Training accuracies: {train_accuracies}\n")
        log_file.write(f"Validation accuracies: {val_accuracies}\n")

    print(f"Metrics saved in log file: {log_file_path}")
