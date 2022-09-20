import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot

import argparse
import logging

from dataset.phyreo import PHYREO
from configs.phyre_cfg import _C as cfg
from utils.predrnn_util import reshape_patch, reshape_patch_back, reserve_schedule_sampling_exp
from Network.predrnn.core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2


def arg_parse():
    parser = argparse.ArgumentParser(description='Pred_PHYRE Parameters')
    parser.add_argument('--pred_model', required=False, type=str, help='Action_cond_predrnn_v2', default='predrnn_v2')
    parser.add_argument('--protocal', required=False, type=str, help='within or cross', default='within')
    parser.add_argument('--fold', required=False, type=int, help='from 0 to 9', default=0)
    parser.add_argument('--epoch', type=int, help='epoch', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--start_epoch', type=int, help='start_epoch', default=0)
    parser.add_argument('--save_interval', type=int, help='save model after how many epoch', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--patch_size', type=int, help='patch size for predRNN', default=8)
    parser.add_argument('--total_length', type=int, help='the gt and the prediction frames number', default=cfg.INPUT.INPUT_SIZE + cfg.NET.PRED_SIZE)
    parser.add_argument('--input_length', type=int, default=cfg.INPUT.INPUT_SIZE)
    parser.add_argument('--pred_path', type=str, default='')
    return parser.parse_args()


args = arg_parse()

# saving and logging
exp_dir = f'results/{args.protocal}{args.fold}_' \
          f'{args.pred_model}_{args.total_length}'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
save_path = f'./{exp_dir}/{args.protocal}{args.fold}'

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'{exp_dir}/exp.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

hp = {"input_length": args.input_length, "total_length": args.total_length,
      "epoch": args.epoch, "batch_size": args.batch_size, "lr": args.lr,
      "save_path": save_path}
logging.info(hp)

# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# model
print(f"Loading prediction model...")
num_hidden = '224,224,224,224'
num_hidden = [int(x) for x in num_hidden.split(',')]
num_layers = len(num_hidden)
pred_model = eval(args.pred_model + '.RNN')(num_layers,num_hidden,args)
pred_model.to(device)

if(args.start_epoch != 0):
    print(f"Resuming from {args.pred_path}")
    pred_model.load_state_dict(torch.load(f'{args.pred_path}'))
else:
    raise ValueError(f"No such model {args.pred_model}")

# optimization
opt = torch.optim.Adam([{'params': pred_model.parameters()}], lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
pixel_loss_nn = nn.MSELoss().cuda()

# dataset
print(f"Creating PHYRE train dataset of {args.protocal}{args.fold} with a batch size of {args.batch_size}")
train_set = PHYREO(split='train')
kwargs = {'pin_memory': True, 'num_workers': 16}
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, **kwargs,
)
print(f"using {len(train_loader.dataset)} training data")

# train
epoch_loss = []
best_loss = 100.
itr = 0
for i in range(args.epoch):
    if(i < args.start_epoch):
        scheduler.step()
        continue 
    sum_loss = []
    all_acc = []
    for batch_idx, (data, label) in enumerate(train_loader):
        itr += 1
        opt.zero_grad()

        data = data.to(device)
        label = label.to(device)
        label_one_hot = one_hot(label.to(torch.int64), 2).float().to(device)
        data = data.permute(0, 1, 3, 4, 2)
        ini = data[:, 0, :].unsqueeze(1)
        data = reshape_patch(data, args.patch_size)
        real_input_flag = reserve_schedule_sampling_exp(itr, data.shape[0], args).float().to(device)

        future_imgs, loss = pred_model(data, real_input_flag)#FIXME:

        loss.backward()
        opt.step()
        scheduler.step()

        sum_loss.append(loss.cpu().detach().numpy())
        print(f'epoch {i} batch {batch_idx} loss: {loss:.4f}')

    mean_loss = np.mean(sum_loss)
    info = f"#######  epoch {i} : mean_loss {mean_loss}  #########"
    print(info)
    logging.info(info)

    # save the model parameters after an interval
    if i % args.save_interval == 0:
        torch.save(pred_model.state_dict(), save_path + f'_pred_model{i+1}.pt')
        if mean_loss < best_loss:
            best_loss = mean_loss

print(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
logging.info(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
