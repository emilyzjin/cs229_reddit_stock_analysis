import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import matplotlib
from args import get_train_args
from collections import OrderedDict
from json import dumps
from model import linearRegression
from ujson import load as json_load
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util
from util import collate_fn
import sys
import os


input_size =
output_size =
lr = 1e-4
weight_decay = 0
num_train =
num_dev =
num_test =
step =
num_epochs =

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # # Get embeddings
    # log.info('Loading embeddings...')
    # word_vectors = util.torch_from_json(args.word_emb_file)
    # char_vectors = util.torch_from_json(args.char_emb_file)

    # set up Dataset object for (train / val / test)

    # Get model
    model = linearRegression(input_size=input_size,
                             output_size=output_size)
    # model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    model = model.to(device)
    model.train()
    # ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = # dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              sampler=sampler.SubsetRandomSampler(range(num_train)))
    dev_dataset = # dataset
    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            sampler=sampler.SubsetRandomSampler(range(num_train, num_train+num_dev)))

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for x_train, y_train in train_loader:
                # Setup for forward
                # cw_idxs = cw_idxs.to(device)
                # qw_idxs = qw_idxs.to(device)
                # cc_idxs = cc_idxs.to(device)
                # qc_idxs = qc_idxs.to(device)
                batch_size = #cw_idxs.size(0)
                optimizer.zero_grad()

                # forward
                outputs = model(x_train)

                # loss
                loss = criterion(outputs, y_train)
                loss_val = loss.item()

                # backward
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                # ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    # ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    # ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                        util.visualize(tbx,
                                       pred_dict=pred_dict,
                                       eval_path=args.dev_eval_file,
                                       step=step,
                                       split='dev',
                                       num_visuals=args.num_visuals)
                    # del cw_idxs
                    # del qw_idxs
                    # del cc_idxs
                    # del qc_idxs
                    # del y1
                    # del y2
                    del outputs
                    torch.cuda.empty_cache()


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()
    criterion = torch.nn.MSELoss()


    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for x_eval, y_eval in data_loader:
            # Setup for forward
            batch_size = cw_idxs.size(0)

            # Forward
            outputs = model(x_eval)
            loss = criterion(outputs, y_eval)
            loss_val = loss.item()
            nll_meter.update(loss.item(), batch_size)

            # # Get F1 and EM scores
            # p1, p2 = log_p1.exp(), log_p2.exp()
            # starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    # if use_squad_v2:
    #     results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())