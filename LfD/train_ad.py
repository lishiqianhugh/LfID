import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot

import argparse
import logging

from dataset.phyreo import PHYREO
from utils.predrnn_util import reshape_patch, reshape_patch_back, reserve_schedule_sampling_exp
from Network.timesformer.models.vit import TimeSformer
from configs.phyre_cfg import _C as cfg
from Network.predrnn import predrnn_v2

import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='Pred_TSF_PHYRE Parameters')
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--pred_model', required=False, type=str, help='predrnn_v2', default='predrnn_v2')
    parser.add_argument('--model_name', required=False, type=str, help='use which cls model', default='TimeSFormer')
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
    parser.add_argument('--pred_resumed', type=bool, default=False)
    parser.add_argument('--pred_path', type=str,default='')
    parser.add_argument('--pixel_loss_weight', type=float, default=1.0)
    parser.add_argument('--seq_loss_weight', type=float, default=1.0)
    parser.add_argument('--img_width', type=int, default=cfg.INPUT.INPUT_WIDTH)
    return parser.parse_args()


args = arg_parse()

# saving and logging
if args.pred_resumed:
    exp_dir = f'results/serial_{args.protocal}{args.fold}_' \
              f'{args.pred_model}_{args.model_name}_{args.total_length}'
else:
    exp_dir = f'results/parallel_{args.protocal}{args.fold}_' \
              f'{args.pred_model}_{args.model_name}_{args.total_length}'

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
    args.device = 'cuda'
else:
    args.device = 'cpu'

# model
print(f"Loading prediction model...")
num_hidden = '224,224,224,224'
num_hidden = [int(x) for x in num_hidden.split(',')]
num_layers = len(num_hidden)
pred_model = eval(args.pred_model + '.RNN')(num_layers, num_hidden, args)
pred_model.to(args.device)

if(args.pred_resumed):
    print(f"Resuming prediction model")
    pred_model.load_state_dict(torch.load(f'{args.pred_path}'))

print(f"Loading TimeSformer...")
pretrained_path = 'Network/timesformer/pretrained/TimeSformer_divST_8x32_224_K600.pyth'
model = TimeSformer(img_size=224, num_classes=2, num_frames=8,
                    attention_type='divided_space_time',
                    pretrained_model=pretrained_path)
model.to(args.device)

# optimization
if(args.pred_resumed):
    opt = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)
else:
    opt = torch.optim.Adam([{'params': pred_model.parameters()},
                            {'params': model.parameters()}], lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
loss_fn = nn.BCELoss()

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
for i in range(args.start_epoch, args.epoch):
    sum_loss = []
    sum_seq_loss = []
    sum_pixel_loss = []
    all_acc = []
    for batch_idx, (data, label) in enumerate(train_loader):
        itr += 1
        opt.zero_grad()
        data = data.to(args.device)
        label = label.to(args.device)
        label_one_hot = one_hot(label.to(torch.int64), 2).float().to(args.device)
        data = data.permute(0, 1, 3, 4, 2)
        ini = data[:, 0, :].unsqueeze(1)
        data = reshape_patch(data, args.patch_size)
        real_input_flag = reserve_schedule_sampling_exp(itr, data.shape[0], args).float().to(args.device)
        future_imgs, pixel_loss = pred_model(data,real_input_flag)
        future_imgs = reshape_patch_back(future_imgs, args.patch_size)
        pdb.set_trace()
        pred_imgs = torch.cat((ini, future_imgs), dim=1).permute((0, 4, 1, 2, 3))

        out = model(pred_imgs.to(args.device)).squeeze(dim=-1)
        out = nn.Softmax(1)(out)
        pred = torch.argmax(out, dim=-1).float()
        acc = (pred == label).sum() / args.batch_size
        all_acc.append(acc.cpu().detach().numpy())
        seq_loss = loss_fn(out, label_one_hot)
        if(args.pred_resumed == False):
            loss = args.seq_loss_weight * seq_loss + args.pixel_loss_weight * pixel_loss
        else:
            loss = seq_loss
        
        loss.backward()
        opt.step()
        scheduler.step()
        sum_loss.append(loss.cpu().detach().numpy())
        sum_seq_loss.append(seq_loss.cpu().detach().numpy())
        sum_pixel_loss.append(pixel_loss.cpu().detach().numpy())
        print(f'epoch {i} batch {batch_idx} acc: {acc.cpu().detach().numpy():.3f} loss: {loss:.4f}')

    mean_loss = np.mean(sum_loss)
    mean_seq_loss = np.mean(sum_seq_loss)
    mean_pixel_loss = np.mean(sum_pixel_loss)
    mean_acc = np.mean(all_acc)
    info = f"#######  epoch {i} : mean_loss {mean_loss} mean_acc {mean_acc} #########"
    print(info)
    logging.info(info)

    # save the model parameters after an interval
    if i % args.save_interval == 0:
        torch.save(model.state_dict(), save_path + f'_{i+1}.pt')
        torch.save(pred_model.state_dict(), save_path + f'_pred_model{i+1}.pt')
        if mean_loss < best_loss:
            best_loss = mean_loss

print(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
logging.info(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
