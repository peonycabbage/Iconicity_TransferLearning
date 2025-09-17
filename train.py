import numpy as np
from sklearn.metrics import confusion_matrix
from utils import AverageMeter

from model import MLP
from dataset import loadedDataset

import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.hub
torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, utils

# -------------------- Device --------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("working on gpu")
else:
    device = torch.device("cpu")
    print("working on cpu")

# -------------------- Args ----------------------
parser = argparse.ArgumentParser(description='Training')
# New: dataset dirs
parser.add_argument('--train-dir', default='./data/train', type=str,
                    help='path to training data root')
parser.add_argument('--val-dir', default='./data/val', type=str,
                    help='path to validation/test data root')

# New: output model root + run subdir
parser.add_argument('--model', default='./save_model', type=str,
                    help='root folder to save models/logs')
parser.add_argument('--run-name', default='chinese8KP', type=str,
                    help='subfolder under --model for this run')

# New: path to a pretrained checkpoint (from another SL dataset)
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint .pth.tar (optional)')

# (kept) model/optim/training args
parser.add_argument('--arch', default='MLP', help='model architecture')
parser.add_argument('--rnn-layers', default=1, type=int,
                    help='number of rnn layers')
parser.add_argument('--hidden-size', default=3000, type=int,
                    help='output size of RNN hidden layers')
parser.add_argument('--fc-size', default=2000, type=int,
                    help='size of fully connected layer before rnn')
parser.add_argument('--epochs', default=2000, type=int,
                    help='manual epoch number')
parser.add_argument('--lr', default=1e-05, type=float,
                    help='initial learning rate')
parser.add_argument('--lr-step', default=2000, type=float,
                    help='learning rate decay frequency')
parser.add_argument('--batch-size', default=32,
                    type=int, help='mini-batch size')
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers')
args = parser.parse_args()

# Build paths
RUN_DIR = os.path.join(args.model, args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)

# TensorBoard: write under this run
writer = SummaryWriter(log_dir=os.path.join(RUN_DIR, "tb"))

# -------------------- I/O helpers ----------------
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save last & (optionally) best checkpoint into the run folder."""
    ckpt_path = os.path.join(RUN_DIR, filename)
    torch.save(state, ckpt_path)
    if is_best:
        best_name = f'{args.run_name}_best.pth.tar'
        shutil.copyfile(ckpt_path, os.path.join(RUN_DIR, best_name))


def adjust_learning_rate(optimizer, epoch):
    """Your original step decay based on --lr-step."""
    if not epoch % args.lr_step and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

# -------------------- Metrics --------------------
y_pred = []
y_true = []
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.cuda()
    target = target.cuda()
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return(res)

# -------------------- Train / Val ----------------
def train(train_loader, model, criterion, optimizer, epoch):
    training_losses = AverageMeter()
    toptrain1 = AverageMeter()
    top5 = AverageMeter()
    model.to(device)
    model.train()   # switch to train mode
    loader_iter = iter(train_loader)

    for i, (inputs, target, _) in enumerate(loader_iter):
        input_var = [input.cuda() for input in inputs]
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        output = output[:, -1, :]
        loss = criterion(output, target_var)
        training_losses.update(loss.item(), 1)

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
        toptrain1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute gradient
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}][{1}/{2}]\t'
              'lr {lr:.5f}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Top1 {toptrain1.val:.3f} ({toptrain1.avg:.3f})\t'
              'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  epoch, i, len(train_loader),
                  lr=optimizer.param_groups[-1]['lr'],
                  loss=training_losses,
                  toptrain1=toptrain1,
                  top5=top5))
        # NOTE: keeping your original prediction line intact
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        target_var = target_var.data.cpu().numpy()
    return(training_losses.avg, toptrain1.avg)


def validate(val_loader, model, criterion):
    val_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.to(device)
    model.eval()

    for i, (inputs, target, _) in enumerate(val_loader):
        input_var = [input.cuda() for input in inputs]
        target_var = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input_var)
            output = output[:, -1, :]
            loss = criterion(output, target_var)
            val_losses.update(loss.item(), 1)

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)

        print('Test: [{0}/{1}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  i, len(val_loader),
                  loss=val_losses,
                  top1=top1,
                  top5=top5))

        # NOTE: keeping original prediction line intact
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        target_var = target_var.data.cpu().numpy()
        y_pred.extend(output)
        y_true.extend(target)
    return (top1.avg, top5.avg, val_losses.avg, y_true, y_pred)

# -------------------- Main -----------------------
if __name__ == '__main__':
    # Data Transform and data loading (kept)
    traindir = args.train_dir
    valdir = args.val_dir

    transform = (transforms.Compose([
                                    transforms.ToTensor()
                                   ]
                                    ),
                 transforms.Compose([
                     transforms.ToTensor()]
    )
    )

    train_dataset = loadedDataset(traindir)
    val_dataset = loadedDataset(valdir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # ----------------- Model init / load -----------------
    ckpt_file = os.path.join(RUN_DIR, 'checkpoint.pth.tar')

    if os.path.exists(ckpt_file):
        # Resume from this run's checkpoint
        model_info = torch.load(ckpt_file, map_location='cpu')
        print("==> resuming existing run '{}' ".format(model_info.get('arch', args.arch)))
        model = MLP(
            model_info['num_classes'], model_info['rnn_layers'], model_info['hidden_size'], model_info['fc_size'])
        model.cuda()
        model.load_state_dict(model_info['state_dict'])
        best_prec = model_info.get('best_prec', 0)
        cur_epoch = model_info.get('epoch', 0)
    elif args.pretrained and os.path.exists(args.pretrained):
        # Load PRETRAINED weights from another sign language (strict=False)
        print(f"==> loading PRETRAINED weights from: {args.pretrained}")
        model = MLP(
            len(train_dataset.classes), args.rnn_layers, args.hidden_size, args.fc_size)
        model.cuda()
        pretrained = torch.load(args.pretrained, map_location='cpu')
        missing, unexpected = model.load_state_dict(pretrained['state_dict'], strict=False)
        print(f"Loaded pretrained with strict=False. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        best_prec = 0
        cur_epoch = 0
    else:
        # Fresh init
        print("==> creating model '{}' ".format(args.arch))
        model = MLP(
            len(train_dataset.classes), args.rnn_layers, args.hidden_size, args.fc_size)
        model.cuda()
        best_prec = 0
        cur_epoch = 0

    # ----------------- Loss / Optim -----------------
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{'params': model.fc_pre.parameters()},
                                  {'params': model.rnn.parameters()},
                                  {'params': model.fc.parameters()}],
                                  lr=args.lr)

    # ----------------- Logging file -----------------
    results_txt = os.path.join(RUN_DIR, 'results.txt')
    epochtrain_loss=[]
    epochval_loss=[]
    epochtrain_accuracy=[]
    epochval_accuracy=[]
    with open(results_txt, 'a+', encoding='utf-8') as myfile:
        myfile.writelines('arch '+str(args.arch) + '\n'
                          'rnn_layers '+str(args.rnn_layers) + '\n'
                          'hidden_size '+str(args.hidden_size) + '\n'
                          'fc_size '+str(args.fc_size) + '\n'
                          'lr '+str(args.lr) + '\n'
                          'lr rate '+str(args.lr_step) + '\n'
                          'total # of epochs '+str(args.epochs) + '\n'
                          'batch size ' +str(args.batch_size) + '\n')

        # ----------------- Train loop -----------------
        for epoch in range(cur_epoch, args.epochs):

            optimizer = adjust_learning_rate(optimizer, epoch)

            print("---------------------------------------------------Training---------------------------------------------------")

            # train on one epoch
            training_losses, toptrain1 = train(train_loader, model,
                               criterion, optimizer, epoch)

            epochtrain_loss.append(np.array(training_losses).mean())
            epochtrain_accuracy.append(np.array(toptrain1).mean())

            print("--------------------------------------------------Validation--------------------------------------------------")

            # evaluate on validation set
            prec1, prec5, val_losses, y_true, y_pred = validate(val_loader, model, criterion)
            epochval_loss.append(np.array(val_losses).mean())
            epochval_accuracy.append(np.array(prec1).mean())

            print("------Validation Result------")
            print("      Top1 accuracy: {prec: .2f} %".format(prec=prec1))
            print("      Top5 accuracy: {prec: .2f} %".format(prec=prec5))
            print("-----------------------------")

            # remember best top1 accuracy and save checkpoint
            is_best = prec1 > best_prec
            best_prec = max(prec1, best_prec)

            save_checkpoint({
                'epoch': epoch + 1,
                'best_prec': best_prec,
                'epochtrain_loss': training_losses,
                'epochval_loss': val_losses,
                'epochtrain_accuracy': toptrain1,
                'epochval_accuracy': prec1,
                'arch': args.arch,
                'num_classes': len(train_dataset.classes),
                'rnn_layers': args.rnn_layers,
                'hidden_size': args.hidden_size,
                'fc_size': args.fc_size,
                'lr rate': args.lr,
                'lr step': args.lr_step,
                'batch size': args.batch_size,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best)

            myfile.writelines("Best prediction "+str(best_prec) + " at epoch " +str(epoch)+'\n')

            # tensorboard
            writer.add_scalar("train loss", training_losses, epoch)
            writer.add_scalar("validation loss", val_losses, epoch)
            writer.add_scalar("train accuracy", toptrain1, epoch)
            writer.add_scalar("validation accuracy", prec1, epoch)

    writer.flush()
