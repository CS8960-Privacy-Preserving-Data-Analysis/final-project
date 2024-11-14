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
from torch.func import grad_and_value, vmap
from augmented_grad_samplers import AugmentationMultiplicity
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torchvision.transforms as transforms
from utils import (init_distributed_mode,initialize_exp,bool_flag,accuracy,get_noise_from_bs,get_epochs_from_bs,print_params,)
from privacy_engine_augmented import PrivacyEngineAugmented
from prepare_models import prepare_data_cifar, prepare_augmult_cifar
from EMA_without_class import create_ema, update
import json




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

best_prec1 = 0

# Pass Optimizer based on Argument
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
    args = parse_args()

    init_distributed_mode(args)
    logger = initialize_exp(args)

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
    # print_params(model)

    rank = args.global_rank
    is_main_worker = rank == 0

    # Modify the model to use GroupNorm instead of BatchNorm since BatchNorm is not supported in DP-SGD
    print("Replacing BatchNorm with GroupNorm for differential privacy...")
    model = ModuleValidator.fix(model)

    # # This is required for Opacus to calculate per-sample gradients
    # model = GradSampleModule(model)

    # # Validate the model for DP compatibility
    # errors = ModuleValidator.validate(model, strict=True)
    # if errors:
    #     print("Model still has unsupported layers:", errors)
    # else:
    #     print("Model is now compatible with differential privacy.")

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
    
    # Get the data loaders
    train_dataset,train_loader, test_loader = prepare_data_cifar(args.data_root,args.batch_size,args.proportion)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    print("Loss function defined.")

    if args.half:
        model.half()
        criterion.half()

    optimizer = choose_optimizer(args, model)
    print("Optimizer initialized.")

    # Create a privacy engine
    #privacy_engine = PrivacyEngine()
    privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)
    sigma = get_noise_from_bs(args.batch_size, args.ref_noise, args.ref_B)

    ##We use our PrivacyEngine Augmented to take into accoung the eventual augmentation multiplicity
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=args.max_grad_norm,
        poisson_sampling=args.poisson_sampling,
        K=args.transform
    )
    print(f"Model made private with target epsilon: {args.target_epsilon}")

    ## Changes the grad samplers to work with augmentation multiplicity
    prepare_augmult_cifar(model,args.transform)
    ema = None

    # we create a shadow model
    print("shadowing de model with EMA")
    ema = create_ema(model)
    train_acc,test_acc,epsilons,losses,top1_accs,grad_sample_gradients_norms_per_step = (0, 0, [], [], [], [])
    norms2_before_sigma = []

    E = get_epochs_from_bs(args.batch_size, args.ref_nb_steps, len(train_dataset))
    if is_main_worker: print(f"E:{E},sigma:{sigma}, BATCH_SIZE:{args.batch_size}, noise_multiplier:{sigma}, EPOCHS:{E}")
    nb_steps = 0
    for epoch in range(E):
        if nb_steps >= args.ref_nb_steps:
            break
        nb_steps, norms2_before_sigma = train(model, ema, train_loader, optimizer, epoch, args.ref_nb_steps, rank, privacy_engine, 
                                                args.transform, logger, losses, top1_accs, epsilons, grad_sample_gradients_norms_per_step, 
                                                val_loader, is_main_worker, args, norms2_before_sigma, nb_steps)

        if is_main_worker:
            print(f"epoch:{epoch}, Current loss:{losses[-1]:.2f},nb_steps:{nb_steps}, top1_acc of model (not ema){top1_accs[-1]:.2f},average gradient norm:{grad_sample_gradients_norms_per_step[-1]:.2f}")

    if is_main_worker:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        test_acc, train_acc = test(ema, val_loader, train_loader, rank)
        print(f"train_acc (EMA):{train_acc:.2f}, test_acc (EMA):{test_acc:.2f}, epsilon:{epsilons[-1]:.2f}")
        logger.info("__log:"+ json.dumps({
                    "final_train_acc_ema": train_acc,
                    "final_test_acc_ema": test_acc,
                    "final_epsilon": epsilons[-1],
                    "avergage_grad_sample_gradients_norms": np.mean(
                        grad_sample_gradients_norms_per_step)
                }))
        test_acc, train_acc = test(model, test_loader, train_loader, rank)
        print(f"final test acc of non ema model:{test_acc:.2f}, final train acc of non ema model:{train_acc:.2f}")
        logger.info("__log:"+ json.dumps({
                    "final_train_acc_non_ema": train_acc,
                    "final_test_acc_non_ema": test_acc,
                }))

def train(model, ema, train_loader, optimizer, epoch, max_nb_steps, device, privacy_engine, K, logger,
    losses, train_acc, epsilons, grad_sample_gradients_norms_per_epoch, test_loader, is_main_worker, args,
    norms2_before_sigma, nb_steps
):
    """
    Trains the model for one epoch. If it is the last epoch, it will stop at max_nb_steps iterations.
    If the model is being shadowed for EMA, we update the model at every step.
    """
    # nb_steps = nb_steps
    model.train()
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    if is_main_worker:print(f"steps_per_epoch:{steps_per_epoch}")
    losses_epoch, train_acc_epoch, grad_sample_norms = [], [], []
    nb_examples_epoch = 0
    max_physical_batch_size_with_augnentation = (args.max_physical_batch_size if K == 0 else args.max_physical_batch_size // K)
    with BatchMemoryManager(data_loader=train_loader,max_physical_batch_size=max_physical_batch_size_with_augnentation,optimizer=optimizer) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            nb_examples_epoch+=len(images)
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            target = target.to(device)
            assert K == args.transform
            l = len(images)
            ##Using Augmentation multiplicity
            if K:
                images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
                target = torch.repeat_interleave(target, repeats=K, dim=0)
                transform = transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),transforms.RandomHorizontalFlip(p=0.5),])
                images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)
                assert len(images) == args.transform * l

            # compute output
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)
            losses_epoch.append(loss.item())
            train_acc_epoch.append(acc)

            loss.backward()
            is_updated = not (optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch

            ## Logging gradient statistics on the main worker
            if is_main_worker:
                per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.grad_sample is not None]
                per_sample_norms = (torch.stack(per_param_norms, dim=1).norm(2, dim=1).cpu().tolist())
                grad_sample_norms += per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s

        
            optimizer.step()
            if is_updated:
                nb_steps += 1  # ?
                if ema:
                    update(model, ema, nb_steps)
                if is_main_worker:
                    losses.append(np.mean(losses_epoch))
                    train_acc.append(np.mean(train_acc_epoch))
                    grad_sample_gradients_norms_per_epoch.append(np.mean(grad_sample_norms))
                    losses_epoch, train_acc_epoch = [],[]
                    if nb_steps % args.freq_log == 0:
                        print(f"epoch:{epoch},step:{nb_steps}")
                        m2 = max(np.mean(norms2_before_sigma)-1/args.batch_size,0)
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "nb_steps":nb_steps,
                                    "train_acc": np.mean(train_acc[-args.freq_log :]),
                                    "loss": np.mean(losses[-args.freq_log :]),
                                    "grad_sample_gradients_norms": np.mean(grad_sample_norms),
                                    "grad_sample_gradients_norms_lowerC": np.mean(np.array(grad_sample_norms)<args.max_per_sample_grad_norm),
                                    #"norms2_before_sigma":list(norms2_before_sigma),
                                   # "grad_sample_gradients_norms_hist":list(np.histogram(grad_sample_norms,bins=np.arange(100), density=True)[0]),
                                }
                            )
                        )
                        norms2_before_sigma=[]
                        grad_sample_norms = []
                    if nb_steps % args.freq_log_val == 0:
                        test_acc_ema, train_acc_ema = (
                            test(ema, test_loader, train_loader, device)
                            if ema
                            else test(model, test_loader, train_loader, device)
                        )
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "test_acc_ema": test_acc_ema,
                                    "train_acc_ema": train_acc_ema,
                                }
                            )
                        )
                nb_examples_epoch=0
                if nb_steps >= max_nb_steps:
                    break
        epsilon = privacy_engine.get_epsilon(args.delta)
        if is_main_worker:
            epsilons.append(epsilon)
        return nb_steps, norms2_before_sigma


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

def test(model, test_loader, train_loader, device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = []
    train_top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            test_top1_acc.append(acc)

    test_top1_avg = np.mean(test_top1_acc)

    with torch.no_grad():
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            # losses.append(loss.item())
            train_top1_acc.append(acc)
    train_top1_avg = np.mean(train_top1_acc)
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")
    return (test_top1_avg, train_top1_avg)

def parse_args():
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

    parser.add_argument("--max_physical_batch_size",default=128,type=int,help="max_physical_batch_size for BatchMemoryManager",)


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

    parser.add_argument("--ref_noise",type=float,default=3,help="reference noise used with reference batch size and number of steps to create our physical constant",)
    parser.add_argument("--ref_B",type=int,default=4096,help="reference batch size used with reference noise and number of steps to create our physical constant",)
    parser.add_argument("--nb_groups",type=int,default=16,help="number of groups for the group norms",)
    parser.add_argument("--ref_nb_steps",default=2500,type=int,help="reference number of steps used with reference noise and batch size to create our physical constant",)

    parser.add_argument("--dump_path",type=str,default="",help="Where results will be stored",)
    parser.add_argument("--transform",type=int,default=0,help="using augmentation multiplicity",)
    parser.add_argument("--freq_log", type=int, default=20, help="every each freq_log steps, we log",)
    parser.add_argument("--freq_log_val",type=int,default=100,help="every each freq_log steps, we log val and ema acc",)
    parser.add_argument("--poisson_sampling",type=bool_flag,default=True,help="using Poisson sampling",)
    parser.add_argument("--proportion",default=1,type=float,help="proportion of the training set to use for training",)
    parser.add_argument("--exp_name", type=str, default="bypass")
    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--order1", type=int, default=0)
    parser.add_argument("--order2", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)
    parser.add_argument("--data_root",type=str,default="",help="Where CIFAR10 is/will be stored",)




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



    return parser.parse_args()

if __name__ == '__main__':
    main()
