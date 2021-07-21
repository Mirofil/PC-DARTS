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

parser.add_argument('--steps_per_epoch', type=float, default=None, help='weight decay for arch encoding')

parser.add_argument('--higher_method' ,       type=str, choices=['val', 'sotl'],   default='sotl', help='Whether to take meta gradients with respect to SoTL or val set (which might be the same as training set if they were merged)')
parser.add_argument('--higher_params' ,       type=str, choices=['weights', 'arch'],   default='arch', help='Whether to do meta-gradients with respect to the meta-weights or architecture')
parser.add_argument('--higher_order' ,       type=str, choices=['first', 'second', None],   default="second", help='Whether to do meta-gradients with respect to the meta-weights or architecture')
parser.add_argument('--higher_loop' ,       type=str, choices=['bilevel', 'joint'],   default="bilevel", help='Whether to make a copy of network for the Higher rollout or not. If we do not copy, it will be as in joint training')
parser.add_argument('--higher_reduction' ,       type=str, choices=['mean', 'sum'],   default='sum', help='Reduction across inner steps - relevant for first-order approximation')
parser.add_argument('--higher_reduction_outer' ,       type=str, choices=['mean', 'sum'],   default='sum', help='Reduction across the meta-betach size')
parser.add_argument('--meta_algo' ,       type=str, choices=['reptile', 'metaprox', 'darts_higher', "gdas_higher", "setn_higher", "enas_higher"],   default=None, help='Whether to do meta-gradients with respect to the meta-weights or architecture')
parser.add_argument('--inner_steps', type=int, default=100, help='Steps for inner loop of bilevel')


parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
parser.add_argument('--perturb_alpha', type=str, default=None, help='portion of training data')
parser.add_argument('--hessian', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True,
                    help='Warm start one-shot model before starting architecture updates.')
args = parser.parse_args()

args.save = 'search-higher-{}-{}'.format(args.save, args.seed)

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


    if args.perturb_alpha:
        epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
        logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
    else:
        epsilon_alpha = None

    # training
    train_acc, train_obj, hypergrads_info = train(train_queue=train_queue, valid_queue=valid_queue, network=model, architect=architect, criterion=criterion, 
                                    w_optimizer=optimizer, a_optimizer=architect.optimizer, epoch=epoch, inner_steps=args.inner_steps, logger=logger,
                                    perturb_alpha=utils.Random_alpha, epsilon_alpha=epsilon_alpha)
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
        
    wandb_log = {"train_acc":train_acc, "train_loss": train_obj, "val_acc": valid_acc, "valid_loss":valid_obj,
                "search.final.cifar10": genotype_perf, 
                "epoch":epoch, "ops":ops_count, "eigval": eigenvalues, "width":width, "depth":depth, "hypergrads":hypergrads_info}
    all_logs.append(wandb_log)
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

  for log in tqdm(all_logs, desc = "Logging WANDB logs as a whole"):
    wandb.log(log)
    

def train(train_queue, valid_queue, network, architect, criterion, w_optimizer, a_optimizer, logger=None, inner_steps=100, 
          epoch=0, steps_per_epoch=None, perturb_alpha=None, epsilon_alpha=None):
    import higher
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    hypergrad_meters = {"first":{"l2":utils.AvgrageMeter(), "cos": utils.AvgrageMeter(), "dot": utils.AvgrageMeter(), "descent": utils.AvgrageMeter()}}

    train_iter = iter(train_queue)
    valid_iter = iter(valid_queue)
    search_loader_iter = zip(train_iter, valid_iter)
    for data_step, ((base_inputs, base_targets), (arch_inputs, arch_targets)) in tqdm(enumerate(search_loader_iter), total = round(len(train_queue)/inner_steps), disable = True if inner_steps < 10 else False):
      if steps_per_epoch is not None and data_step > steps_per_epoch:
        break
      network.train()
      n = base_inputs.size(0)

      base_inputs = base_inputs.cuda()
      base_targets = base_targets.cuda(non_blocking=True)

      # get a random minibatch from the search queue with replacement
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda(non_blocking=True)
      
      all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, 
                                                                                                search_loader_iter, inner_steps=args.inner_steps, args=args)
      weights_mask = [1 if ('arch' not in n and 'alpha' not in n) else 0 for (n, p) in network.named_parameters()] # Zeroes out all the architecture gradients in Higher. It has to be hacked around like this due to limitations of the library
      zero_arch_grads = lambda grads: [g*x if g is not None else None for g,x in zip(grads, weights_mask)]
      monkeypatch_higher_grads_cond = True if (args.meta_algo not in ['reptile', 'metaprox'] and (args.higher_order != "first" or args.higher_method == "val")) else False
      diffopt_higher_grads_cond = True if (args.meta_algo not in ['reptile', 'metaprox'] and args.higher_order != "first") else False
      fnetwork = higher.patch.monkeypatch(network, device='cuda', copy_initial_weights=True if args.higher_loop == "bilevel" else False, track_higher_grads = monkeypatch_higher_grads_cond)
      diffopt = higher.optim.get_diff_optim(w_optimizer, network.parameters(), fmodel=fnetwork, grad_callback=zero_arch_grads, device='cuda', override=None, track_higher_grads = diffopt_higher_grads_cond) 
      fnetwork.zero_grad() # TODO where to put this zero_grad? was there below in the sandwich_computation=serial branch, tbut that is surely wrong since it wouldnt support higher meta batch size
      
      sotl, first_order_grad = [], None
      inner_rollouts, meta_grads = [], [] # For implementing meta-batch_size in Reptile/MetaProx and similar

      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if args.higher_method == "sotl_v2":
              base_inputs, base_targets = arch_inputs, arch_targets ## Use train set for the unrolling to compute hypergradients, then forget the training and train weights only using a separate set
        #   if data_step in [0, 1] and inner_step < 3:
        #       print(f"Base targets in the inner loop at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
          if args.perturb_alpha:
            # print('before softmax', model.arch_parameters())
            fnetwork.softmax_arch_parameters()
            # perturb on alpha
            # print('after softmax', model.arch_parameters())
            perturb_alpha(fnetwork, base_inputs, base_targets, epsilon_alpha)
            diffopt.zero_grad()
            architect.optimizer.zero_grad()
            # print('afetr perturb', model.arch_parameters())
          logits = fnetwork(base_inputs)
          base_loss = criterion(logits, base_targets)
          sotl.append(base_loss)
          new_params, cur_grads = diffopt.step(base_loss)
          cur_grads = list(cur_grads)
          for idx, (g, p) in enumerate(zip(cur_grads, fnetwork.parameters())):
              if g is None:
                  cur_grads[idx] = torch.zeros_like(p)
          first_order_grad_for_free_cond = True
          first_order_grad_concurrently_cond = False
          first_order_grad = fo_grad_if_possible(args, fnetwork, criterion, all_arch_inputs, all_arch_targets, arch_inputs, arch_targets, cur_grads, inner_step, 
                                                  data_step, 1, first_order_grad, first_order_grad_for_free_cond, first_order_grad_concurrently_cond, logger=None)
      meta_grads, inner_rollouts = hypergrad_outer(args=args, fnetwork=fnetwork, criterion=criterion, arch_targets=arch_targets, arch_inputs=arch_inputs,
                                          all_arch_inputs=all_arch_inputs, all_arch_targets=all_arch_targets, all_base_inputs=all_base_inputs, all_base_targets=all_base_targets,
                                          sotl=sotl, inner_step=inner_step, inner_steps=inner_steps, inner_rollouts=inner_rollouts,
                                          first_order_grad_for_free_cond=first_order_grad_for_free_cond, first_order_grad_concurrently_cond=first_order_grad_concurrently_cond,
                                          monkeypatch_higher_grads_cond=monkeypatch_higher_grads_cond, zero_arch_grads_lambda=zero_arch_grads, meta_grads=meta_grads,
                                          step=data_step, epoch=epoch, logger=None)
  
      if first_order_grad is not None and args.higher_order != "second":
          assert first_order_grad_for_free_cond or first_order_grad_concurrently_cond
          if epoch < 2:
              print(f"Putting first_order_grad into meta_grads (NOTE we aggregate first order grad by summing in the first place to save memory, so dividing by inner steps gives makes it average over the rollout) (len of first_order_grad ={len(first_order_grad)}, len of param list={len(list(network.parameters()))}) with reduction={args.higher_reduction}, inner_steps (which is the division factor)={inner_steps}, head={first_order_grad[0]}")
          if args.higher_reduction == "sum": # the first_order_grad is computed in a way that equals summing
              meta_grads.append(first_order_grad)
          else:
              meta_grads.append([g/inner_steps if g is not None else g for g in first_order_grad])  
              
      avg_meta_grad = hyper_meta_step(network, inner_rollouts, meta_grads, args, data_step, logger, model_init=None, outer_iters=1, epoch=epoch)
      
      
      if args.higher_order == "second":
        stacked_fo_grad = torch.cat([g.view(-1) for g in first_order_grad]).flatten().cpu().numpy()
        stacked_meta_grad = torch.cat([g.view(-1) for g in avg_meta_grad]).flatten().cpu().numpy()

        hypergrad_meters["first"]["cos"].update(scipy.spatial.distance.cosine(stacked_fo_grad, stacked_meta_grad))
        hypergrad_meters["first"]["l2"].update(np.linalg.norm(stacked_fo_grad-stacked_meta_grad))
        
        dot_product = np.dot(stacked_fo_grad, stacked_meta_grad)
        hypergrad_meters["first"]["dot"].update(dot_product)
        hypergrad_meters["first"]["sign"].update()
      else:
        hypergrad_info = {}
      
      with torch.no_grad():  # Update the pre-rollout weights
          for (n, p), g in zip(network.named_parameters(), avg_meta_grad):
              cond = ('arch' not in n and 'alpha' not in n) if args.higher_params == "weights" else ('arch' in n or 'alpha' in n)  # The meta grads typically contain all gradient params because they arise as a result of torch.autograd.grad(..., model.parameters()) in Higher
              if cond:
                  if g is not None and p.requires_grad:
                      p.grad = g
      a_optimizer.step()
      
      w_optimizer.zero_grad()
      architect.optimizer.zero_grad()
      
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if data_step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
              logger.info(f"Doing weight training for real in higher_loop={args.higher_loop} at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
          if args.perturb_alpha:
            # print('before softmax', model.arch_parameters())
            network.softmax_arch_parameters()
                
            # perturb on alpha
            # print('after softmax', model.arch_parameters())
            perturb_alpha(network, base_inputs, base_targets, epsilon_alpha)
            w_optimizer.zero_grad()
            architect.optimizer.zero_grad()
            # print('afetr perturb', model.arch_parameters())
          logits = network(base_inputs)
          base_loss = criterion(logits, base_targets)
          network.zero_grad()
          base_loss.backward()
          w_optimizer.step()
          n = base_inputs.size(0)

          prec1, prec5 = utils.accuracy(logits, base_targets, topk=(1, 5))

          objs.update(base_loss.item(), n)
          top1.update(prec1.data, n)
          top5.update(prec5.data, n)

      if data_step % args.report_freq == 0:
          logging.info('train %03d %e %f %f', data_step, objs.avg, top1.avg, top5.avg)
      if 'debug' in args.save:
          break

    hypergrad_info = {"cos":hypergrad_meters["cos"].avg, "l2":hypergrad_meters["l2"].avg, "dot": hypergrad_meters["dot"].avg}

    return  top1.avg, objs.avg, hypergrad_info
    

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

