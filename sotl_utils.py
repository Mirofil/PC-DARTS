# Copied from previous adaptations


import torch
import higher
from hessian_eigenthings import compute_hessian_eigenthings
from torch.autograd import Variable

def _hessian(outputs, inputs, out=None, allow_unused=False,
              create_graph=False, weight_decay=3e-5):
    #assert outputs.data.ndimension() == 1

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = Variable(torch.zeros(n, n)).type_as(outputs)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                      allow_unused=allow_unused)
        grad = grad.contiguous().view(-1) + weight_decay*inp.view(-1)
        #grad = outputs[i].contiguous().view(-1)

        for j in range(inp.numel()):
            # print('(i, j): ', i, j)
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True)[j:]
            else:
                n = sum(x.numel() for x in inputs[i:]) - j
                row = Variable(torch.zeros(n)).type_as(grad[j])
                #row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out.data[ai, ai:].add_(row.clone().type_as(out).data)  # ai's row
            if ai + 1 < n:
                out.data[ai + 1:, ai].add_(row.clone().type_as(out).data[1:])  # ai's column
            del row
            ai += 1
        del grad
    return out

def exact_hessian(network, val_loader, criterion, xloader, epoch, logger, args):
  labels = []
  for i in range(network._max_nodes):
    for n in network._op_names:
      labels.append(n + "_" + str(i))

  network.logits_only=True
  val_x, val_y = next(iter(val_loader))
  val_loss = criterion(network(val_x.to('cuda')), val_y.to('cuda'))
  try:
    train_x, train_y, _, _ = next(iter(xloader))
  except:
    train_x, train_y = next(iter(xloader))

  train_loss = criterion(network(train_x.to('cuda')), train_y.to('cuda'))
  val_hessian_mat = _hessian(val_loss, network.arch_params())
  if epoch == 0:
    print(f"Example architecture Hessian: {val_hessian_mat}")
  val_eigenvals, val_eigenvecs = torch.eig(val_hessian_mat)
  try:
    if not args.merge_train_val_supernet:
      train_hessian_mat = _hessian(train_loss, network.arch_params())
      train_eigenvals, train_eigenvecs = torch.eig(train_hessian_mat)
    else:
      train_eigenvals = val_eigenvals
  except:
    train_eigenvals = val_eigenvals
  val_eigenvals = val_eigenvals[:, 0] # Drop the imaginary components
  if epoch == 0:
    print(f"Example architecture eigenvals: {val_eigenvals}")
  train_eigenvals = train_eigenvals[:, 0]
  val_dom_eigenvalue = torch.max(val_eigenvals)
  train_dom_eigenvalue = torch.max(train_eigenvals)
  eigenvalues = {"max":{}, "spectrum": {}}
  eigenvalues["max"]["train"] = train_dom_eigenvalue
  eigenvalues["max"]["val"] = val_dom_eigenvalue
  eigenvalues["spectrum"]["train"] = {k:v for k,v in zip(labels, train_eigenvals)}
  eigenvalues["spectrum"]["val"] = {k:v for k,v in zip(labels, val_eigenvals)}
  network.logits_only=False
  return eigenvalues
    
def approx_hessian(network, val_loader, criterion, xloader, args):
  network.logits_only=True
  val_eigenvals, val_eigenvecs = compute_hessian_eigenthings(network, val_loader, criterion, 1, mode="power_iter", 
                                                             power_iter_steps=50, arch_only=True, full_dataset=True)
  val_dom_eigenvalue = val_eigenvals[0]
  try:
    if hasattr(args, "merge_train_val_supernet") and not args.merge_train_val_supernet:
      train_eigenvals, train_eigenvecs = compute_hessian_eigenthings(network, val_loader, criterion, 1, mode="power_iter", 
                                                                    power_iter_steps=50, arch_only=True, full_dataset=True)
      train_dom_eigenvalue = train_eigenvals[0]
    else:
      train_eigenvals, train_eigenvecs = None, None
      train_dom_eigenvalue = None
  except:
    train_eigenvals, train_eigenvecs, train_dom_eigenvalue = None, None, None
  eigenvalues = {"max":{}, "spectrum": {}}
  eigenvalues["max"]["val"] = val_dom_eigenvalue
  eigenvalues["max"]["train"] = train_dom_eigenvalue
  network.logits_only=False
  network.zero_grad()
  return eigenvalues


def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args, epoch = 1000, loader_type="train-val"):

    base_inputs, base_targets = base_inputs.cuda(non_blocking=True), base_targets.cuda(non_blocking=True)
    arch_inputs, arch_targets = arch_inputs.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    if args.higher_method == "sotl":
        arch_inputs, arch_targets = None, None
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        if args.inner_steps_same_batch and (args.warm_start is None or epoch >= args.warm_start):
            all_base_inputs.append(base_inputs)
            all_base_targets.append(base_targets)
            all_arch_inputs.append(arch_inputs)
            all_arch_targets.append(arch_targets)
            continue # If using the same batch, we should not try to query the search_loader_iter for more samples
        try:
            if loader_type == "train-val" or loader_type == "train-train":
              (extra_base_inputs, extra_base_targets), (extra_arch_inputs, extra_arch_targets)= next(search_loader_iter)
            else:
              extra_base_inputs, extra_base_targets = next(search_loader_iter)
              extra_arch_inputs, extra_arch_targets = None, None
        except Exception as e:
            continue
        # extra_base_inputs, extra_arch_inputs = extra_base_inputs.cuda(non_blocking=True), extra_arch_inputs.cuda(non_blocking=True)
        # extra_base_targets, extra_arch_targets = extra_base_targets.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        extra_base_inputs, extra_base_targets = extra_base_inputs.cuda(non_blocking=True), extra_base_targets.cuda(non_blocking=True)
        if extra_arch_inputs is not None and extra_arch_targets is not None:
          extra_arch_inputs, extra_arch_targets = extra_arch_inputs.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets


import torch
import sys
import os
from copy import deepcopy
from typing import *

def avg_state_dicts(state_dicts: List):
  if len(state_dicts) == 1:
    return state_dicts[0]
  else:
    mean_state_dict = {}
    for k in state_dicts[0].keys():
      mean_state_dict[k] = sum([network[k] for network in state_dicts])/len(state_dicts)
    return mean_state_dict

def fo_grad_if_possible(args, fnetwork, criterion, all_arch_inputs, all_arch_targets, arch_inputs, arch_targets, cur_grads, inner_step, step, outer_iter, first_order_grad, first_order_grad_for_free_cond, first_order_grad_concurrently_cond, logger=None):
    if first_order_grad_for_free_cond: # If only doing Sum-of-first-order-SOTL gradients in FO-SOTL-DARTS or similar, we can just use these gradients that were already computed here without having to calculate more gradients as in the second-order gradient case
        if inner_step < 3 and step == 0:
            msg = f"Adding cur_grads to first_order grads at inner_step={inner_step}, step={step}, outer_iter={outer_iter}. First_order_grad is head={str(first_order_grad)[0:100]}, cur_grads is {str(cur_grads)[0:100]}"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
        with torch.no_grad():
          if first_order_grad is None:
            first_order_grad = cur_grads
          else:
            first_order_grad = [g1 + g2 for g1, g2 in zip(first_order_grad, cur_grads)]
    elif first_order_grad_concurrently_cond:
      # NOTE this uses a different arch_sample everytime!
        if args.higher_method == "val":
          _, logits = fnetwork(all_arch_inputs[len(all_arch_inputs)-1])
          arch_loss = criterion(logits, all_arch_targets[len(all_arch_targets)-1]) * (1 if args.sandwich is None else 1/args.sandwich)
        elif args.higher_method == "val_multiple":
          _, logits = fnetwork(arch_inputs)
          arch_loss = criterion(logits, arch_targets) * (1 if args.sandwich is None else 1/args.sandwich)
        cur_grads = torch.autograd.grad(arch_loss, fnetwork.parameters(), allow_unused=True)
        with torch.no_grad():
          if first_order_grad is None:
            first_order_grad = cur_grads
          else:
            first_order_grad += [g1 + g2 for g1, g2 in zip(first_order_grad, cur_grads)]
    return first_order_grad

def hyper_meta_step(network, inner_rollouts, meta_grads, args, data_step, logger = None, model_init=None, outer_iters=1, epoch=0):

    if args.meta_algo in ["darts_higher", "gdas_higher", "setn_higher"]: assert args.higher_params == "arch"
    if args.meta_algo in ['reptile', 'metaprox']:
        avg_inner_rollout = avg_state_dicts(inner_rollouts)
        avg_meta_grad = [(p - p_init) for p, p_init in zip(avg_inner_rollout.values(), model_init.parameters())]
        if data_step == 0:
            for i, rollout in enumerate(inner_rollouts):
                msg = f"Printing {i}-th rollout's weight sample: {str(list(rollout.values())[1])[0:75]}"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)
            msg = f"Average of all rollouts: {str(list(avg_inner_rollout.values())[1])[0:75]}"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
        network.load_state_dict(
            model_init.state_dict())  # Need to restore to the pre-rollout state before applying meta-update
    else:
        # Sum over outer_iters metagrads - if they were meant to be averaged/summed, it has to be done at the time the grads from inner_iters are put into meta_grads!
        if epoch < 2 and logger is not None:
            msg = f"Reductioning in the outer loop (len(meta_grads)={len(meta_grads)}, head={str(meta_grads)[0:150]}) with outer reduction={args.higher_reduction_outer}, outer_iters={outer_iters}"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
        with torch.no_grad():
            if args.higher_reduction_outer == "sum":
                avg_meta_grad = [sum([g if g is not None else 0 for g in grads]) for grads in zip(*meta_grads)]
            elif args.higher_reduction_outer == "mean":
                avg_meta_grad = [sum([g if g is not None else 0 for g in grads]) / outer_iters for grads in
                                 zip(*meta_grads)]

    # The architecture update itself
    with torch.no_grad():  # Update the pre-rollout weights
        for (n, p), g in zip(network.named_parameters(), avg_meta_grad):
            cond = 'arch' not in n if args.higher_params == "weights" else 'arch' in n  # The meta grads typically contain all gradient params because they arise as a result of torch.autograd.grad(..., model.parameters()) in Higher
            if cond:
                if g is not None and p.requires_grad:
                    p.grad = g
    return avg_meta_grad

def hypergrad_outer(
    args,
    fnetwork,
    criterion,
    arch_targets,
    arch_inputs,
    all_arch_inputs,
    all_arch_targets,
    all_base_inputs,
    all_base_targets,
    sotl,
    inner_step,
    inner_steps,
    inner_rollouts,
    first_order_grad_for_free_cond,
    first_order_grad_concurrently_cond,
    monkeypatch_higher_grads_cond,
    zero_arch_grads_lambda,
    meta_grads,
    step,
    epoch,
    logger=None,
):
    if args.meta_algo in ["reptile", "metaprox"]:
        inner_rollouts.append(deepcopy(fnetwork.state_dict()))
    elif args.meta_algo:
        if args.higher_method.startswith("val"):
            if args.higher_order == "second":
                _, logits = fnetwork(
                    arch_inputs, params=fnetwork.parameters(time=inner_step)
                )
                arch_loss = [criterion(logits, arch_targets)]
                meta_grad = torch.autograd.grad(
                    sum(arch_loss), fnetwork.parameters(time=0), allow_unused=True
                )
                meta_grads.append(meta_grad)
            elif args.higher_order == "first":
                if not (
                    first_order_grad_for_free_cond or first_order_grad_concurrently_cond
                ):  # Computing the val grads concurrently allows to avoid gradient tracking in Higher
                    if args.higher_method == "val":
                        all_logits = [
                            fnetwork(arch_inputs, params=fnetwork.parameters(time=i))[1]
                            for i in range(0, inner_steps)
                        ]
                        arch_loss = [
                            criterion(all_logits[i], arch_targets)
                            for i in range(len(all_logits))
                        ]
                    elif args.higher_method == "val_multiple":
                        all_logits = [
                            fnetwork(
                                all_arch_inputs[i], params=fnetwork.parameters(time=i)
                            )[1]
                            for i in range(0, inner_steps)
                        ]
                        arch_loss = [
                            criterion(all_logits[i], all_arch_targets[i])
                            for i in range(len(all_logits))
                        ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    if step == 0 and epoch < 2:
                        msg = f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                        if logger is not None:
                            logger.info(msg)
                        else:
                            print()
                    if args.higher_reduction == "sum":

                        fo_grad = [sum(grads) for grads in zip(*all_grads)]
                    elif args.higher_reduction == "mean":
                        fo_grad = [
                            sum(grads) / inner_steps for grads in zip(*all_grads)
                        ]
                    meta_grads.append(fo_grad)
                else:
                    pass

        elif args.higher_method == "sotl":
            if args.higher_order == "second":
                meta_grad = torch.autograd.grad(
                    sum(sotl), fnetwork.parameters(time=0), allow_unused=True
                )
                meta_grads.append(meta_grad)

            elif args.higher_order == "first":
                if not (
                    first_order_grad_for_free_cond or first_order_grad_concurrently_cond
                ):  # TODO I think the for_free branch puts each individual FO grad into meta_grads but here we put only average - though shouldnt really make a difference I think since we just sum over them either now or later?
                    all_logits = [
                        fnetwork(
                            all_base_inputs[i], params=fnetwork.parameters(time=i)
                        )[1]
                        for i in range(0, inner_steps)
                    ]
                    arch_loss = [
                        criterion(all_logits[i], all_base_targets[i])
                        for i in range(len(all_logits))
                    ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    if step == 0 and epoch < 2:
                        if logger is not None:
                            logger.info(
                                f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                            )
                            logger.info(f"Grads sample before: {all_grads[0]}")
                        else:
                            print(
                                f"Reductioning all_grads (len={len(all_grads)} with reduction={args.higher_reduction}, inner_steps={inner_steps}"
                            )
                            print(f"Grads sample before: {all_grads[0]}")
                    with torch.no_grad():
                        if args.higher_reduction == "sum":
                            fo_grad = [sum(grads) for grads in zip(*all_grads)]
                        elif args.higher_reduction == "mean":
                            fo_grad = [
                                sum(grads) / inner_steps for grads in zip(*all_grads)
                            ]
                    if step == 0:
                        print(f"Grads sample after: {fo_grad[0]}")
                    meta_grads.append(fo_grad)
                elif step == 0 and monkeypatch_higher_grads_cond:
                    all_logits = [
                        fnetwork(
                            all_base_inputs[i], params=fnetwork.parameters(time=i)
                        )[1]
                        for i in range(0, inner_steps)
                    ]
                    arch_loss = [
                        criterion(all_logits[i], all_base_targets[i])
                        for i in range(len(all_logits))
                    ]
                    all_grads = [
                        torch.autograd.grad(arch_loss[i], fnetwork.parameters(time=i))
                        for i in range(0, inner_steps)
                    ]
                    assert torch.all_close(
                        zero_arch_grads_lambda(all_grads[0]), meta_grads[0]
                    )
                    logger.info(
                        f"Correctnes of first-order gradients was checked! Samples:"
                    )
                    print(all_grads[0][0])
                    print(meta_grads[0][0])
                else:
                    pass
    return meta_grads, inner_rollouts
