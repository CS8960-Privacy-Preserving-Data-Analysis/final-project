import argparse
import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from opacus import PrivacyEngine, GradSampleModule
from opacus.validators import ModuleValidator
from opacus.grad_sample.functorch import make_functional
from torch.func import grad_and_value, vmap


import resnet
from data_loader import get_data_loaders
from lion import Lion
from visualizer import plot_train_test_loss_accuracy_vs_epochs
from utils import accuracy, AverageMeter, save_checkpoint, log_metrics

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--target-epsilon', default=None, type=float,
                    help='Target epsilon for differential privacy')
parser.add_argument('--delta', default=1e-5, type=float,
                    help='Target delta for differential privacy')
parser.add_argument('--noise-multiplier', default=1.1, type=float,
                    help='Noise multiplier for differential privacy (default: 1.1)')
parser.add_argument('--max-grad-norm', default=1.0, type=float,
                    help='Max grad norm for differential privacy (default: 1.0)')

# For RMSprop
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSprop smoothing constnt (default: 0.99)')
parser.add_argument('--rms-epsilon', default=1e-8, type=float,
                    help='RMSprop epsilon (default: 1e-8)')
parser.add_argument('--centered', default=False, type=bool,
                    help='RMSprop centered (default: False)')

# For Adam optimizer
parser.add_argument('--beta1', default=0.9, type=float,
                    help='Beta1 for Adam optimizer')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='Beta2 for Adam optimizer')

# Choose Optimizer Type
parser.add_argument('--optimizer', default='DP-SGD', type=str,
                    help='Choose Optimizer (default: DP-SGD)')

# Choose Optimizer Type
parser.add_argument('--seed', default=100, type=int,
                    help='Choose Random Seed (default:100)')

best_prec1 = 0


def choose_optimizer(args,model):
    if args.optimizer == 'DP-SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'DP-Adam':

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'DP-RMSprop':
       optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.lr,
                                    alpha=args.alpha,
                                    eps=args.rms_epsilon,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum,
                                    centered=args.centered)

    elif args.optimizer == 'DP-Lion':
        optimizer = Lion(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

    else:
        print('Error: Choose Optimizer from DP-SGD, DP-Adam, DP-RMSprop, DP-Lion')
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    print(f"Optimizer: {args.optimizer} initialized.")

    return optimizer

def main():
    global args, best_prec1, train_losses, train_accuracies, val_losses, val_accuracies
    args = parser.parse_args()

    # Let's set a random seed before training to make the experiments reproducible
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Clear global lists before starting a new experiment
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")
        print("path of the directory: ", os.path.abspath(args.save_dir))
    else:
        print(f"Using existing directory: {args.save_dir}")
        print("WARNING: Contents of the directory will be overwritten.")
        print("path of the directory: ", os.path.abspath(args.save_dir))

    print(f"Initializing {args.optimizer} model...")
    model = resnet.__dict__[args.arch]().cuda()

    print(f"Model {args.arch} initialized.")

    # Modify the model to use GroupNorm instead of BatchNorm since BathNorm is not supported in DP-SGD
    print("Replacing BatchNorm with GroupNorm for differential privacy...")
    model = ModuleValidator.fix(model)

    # This is required for Opacus to calculate per-sample gradients
    model = GradSampleModule(model)

    # Validate the model for DP compatibility
    errors = ModuleValidator.validate(model, strict=True)
    if errors:
        print("Model still has unsupported layers:", errors)
    else:
        print("Model is now compatible with differential privacy.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True
    print("CUDNN benchmarking enabled.")

    # Get the data loaders
    train_loader, val_loader = get_data_loaders(args.batch_size, args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    print("Loss function defined.")

    if args.half:
        model.half()
        criterion.half()

    optimizer = choose_optimizer(args, model)
    print("Optimizer initialized.")

    # Create a privacy engine
    privacy_engine = PrivacyEngine()

    # Make the model private with epsilon if target_epsilon is provided, otherwise use epochs
    if args.target_epsilon:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            grad_sample_mode="no_op"
        )
        print(f"Model made private with target epsilon: {args.target_epsilon}")
    else:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"Model made private for {args.epochs} epochs.")

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        print("Evaluating model...")
        validate(val_loader, model, criterion)
        return

    print("Starting training...")

    train_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print('Current learning rate: {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, privacy_engine=privacy_engine)
        lr_scheduler.step()

        # evaluate on validation set
        print("Validating model...")
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    print("Training completed.")

    # TODO: Fix the optimizer name later
    experiment_id = f"experiment_{args.optimizer}_{args.batch_size}_{args.lr}_{int(time.time())}"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_dir = os.path.join(project_root, 'experiments', experiment_id)

    # Ensure the folder exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = list(range(args.start_epoch, args.epochs))

    # Assuming you want to plot the loss and accuracy after the training ends
    plot_train_test_loss_accuracy_vs_epochs(epochs, train_losses, val_losses, train_accuracies, val_accuracies,
                                  args.target_epsilon, save_dir)

    train_end_time = time.time()
    final_epsilon = privacy_engine.get_epsilon(delta=args.delta)

    # Print the metrics
    print(f"Total training time: {train_end_time - train_start_time:.2f} seconds")
    print(f"Best accuracy: {best_prec1:.2f}%")
    print(f"Best privacy budget (ε): {privacy_engine.get_epsilon(delta=args.delta):.2f}, δ: {args.delta}")
    print(f"Model saved at: {os.path.abspath(args.save_dir)}")

    # Save the metrics to a log file
    log_metrics(args, best_prec1, train_end_time - train_start_time, save_dir, experiment_id, train_losses, val_losses,
                train_accuracies, val_accuracies, final_epsilon)


def train(train_loader, model, criterion, optimizer, epoch, privacy_engine=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        import pdb;pdb.set_trace()
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    # Store the average loss and accuracy for this epoch
    train_losses.append(losses.avg)
    train_accuracies.append(top1.avg)

    # After the epoch, if privacy is enabled, report privacy budget
    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        print(f"Epoch {epoch + 1} | Privacy budget (ε): {epsilon:.2f}, δ: {args.delta}")


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # Store the average loss and accuracy for this epoch
    val_losses.append(losses.avg)
    val_accuracies.append(top1.avg)

    return top1.avg

if __name__ == '__main__':
    main()
