import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import utils as utils
import wandb
import nasbench301 as nb
from genotypes import count_ops
from pathlib import Path
from tqdm import tqdm
from utils import genotype_depth, genotype_width
from visualize import plot
from sotl_utils import format_input_data, fo_grad_if_possible, hyper_meta_step, hypergrad_outer, approx_hessian, exact_hessian

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')


parser.add_argument('--hessian', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True,
                    help='Warm start one-shot model before starting architecture updates.')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, args.seed)

try:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
except Exception as e:
  print(f"Couldnt create exp dir due to {e}")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def get_torch_home():
    if "TORCH_HOME" in os.environ:
        return os.environ["TORCH_HOME"]
    elif os.path.exists('/storage/.torch/'):
        return os.path.join('/storage/', ".torch")

    elif "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".torch")
    else:
        raise ValueError(
            "Did not find HOME in os.environ. "
            "Please at least setup the path of HOME or TORCH_HOME "
            "in the environment."
        )
        
def wandb_auth(fname: str = "nas_key.txt"):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()
  
def load_nb301():
    version = '0.9'
    current_dir = os.path.dirname(get_torch_home())
    models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # If the models are not available at the paths, automatically download
    # the models
    # Note: If you would like to provide your own model locations, comment this out
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                        download_dir=current_dir)

    # Load the performance surrogate model
    #NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    #NOTE: Defaults to using the default model download path
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = model_paths['xgb']
    print(ensemble_dir_performance)
    performance_model = nb.load_ensemble(ensemble_dir_performance)
    
    return performance_model


CIFAR_CLASSES = 10
if args.set=='cifar100':
    CIFAR_CLASSES = 100
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  logger = logging.getLogger()

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  
  wandb_auth()
  run = wandb.init(project="NAS", group=f"Search_Cell_pcdarts", reinit=True)
  
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)


  api = load_nb301()

  if os.path.exists(Path(args.save) / "checkpoint.pt"):
    print(f"Loading checkpoint")
    checkpoint = torch.load(Path(args.save) / "checkpoint.pt")
    optimizer.load_state_dict(checkpoint["w_optimizer"])
    architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
    model.load_state_dict(checkpoint["model"])
    scheduler.load_state_dict(checkpoint["w_scheduler"])
    start_epoch = checkpoint["epoch"]
    all_logs = checkpoint["all_logs"]

  else:
    print(f"Path at {Path(args.save) / 'checkpoint.pt'} does not exist")
    start_epoch=0
    all_logs=[]

  for epoch in tqdm(range(start_epoch, args.epochs), desc = "Iterating over epochs"):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    genotype_perf = api.predict(config=genotype, representation='genotype', with_noise=False)
    ops_count = count_ops(genotype)
    width = {k:genotype_width(g) for k, g in [("normal", genotype.normal), ("reduce", genotype.reduce)]}
    depth = {k:genotype_depth(g) for k, g in [("normal", genotype.normal), ("reduce", genotype.reduce)]}
    
    logging.info(f"Genotype performance: {genotype_perf}, width: {width}, depth: {depth}, ops_count: {ops_count}")
    
    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)


    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)


    if False and args.hessian and torch.cuda.get_device_properties(0).total_memory < 20147483648: # doesnt work anywhere
        eigenvalues = approx_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, args=args)
        # eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)
    elif args.hessian and torch.cuda.get_device_properties(0).total_memory > 20147483648:
        eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)
    else:
        eigenvalues = None
        
    wandb_log = {"train_acc":train_acc, "train_loss":train_obj, "val_acc": valid_acc, "valid_loss":valid_obj, "genotype":str(genotype),
                 "search.final.cifar10": genotype_perf, "epoch":epoch, "ops": ops_count, "depth":depth, "width":width, "eigenvalues":eigenvalues}
    wandb.log(wandb_log)
    all_logs.append(wandb_log)
    
    if epoch % 10 == 0 or epoch == args.epochs - 1:
      try:
        plot(genotype.normal, os.path.join(wandb.run.dir, f"normal_epoch{epoch}") )
        plot(genotype.reduce, os.path.join(wandb.run.dir, f"reduce_epoch{epoch}") )
      except:
        print(f"Failed to save visualized genotypes")

    utils.save_checkpoint2({"model":model.state_dict(), "w_optimizer":optimizer.state_dict(), 
                           "a_optimizer":architect.optimizer.state_dict(), "w_scheduler":scheduler.state_dict(), "epoch": epoch, "all_logs":all_logs}, 
                          Path(args.save) / "checkpoint.pt")
    print(f"Saved checkpoint to {Path(args.save) / 'checkpoint.pt'}")

    # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    #try:
    #  input_search, target_search = next(valid_queue_iter)
    #except:
    #  valid_queue_iter = iter(valid_queue)
    #  input_search, target_search = next(valid_queue_iter)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    if epoch>=15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if step > 25:
        break
      #input = input.cuda()
      #target = target.cuda(non_blocking=True)
      input = Variable(input).cuda()
      target = Variable(target).cuda()
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

