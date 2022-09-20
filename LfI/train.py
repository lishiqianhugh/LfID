import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
import timm

import argparse
import logging

from dataset.phyreo import PHYREO


def arg_parse():
    parser = argparse.ArgumentParser(description='LfI_PHYRE Parameters')
    parser.add_argument('--protocal', required=False, type=str, help='within or cross', default='within')
    parser.add_argument('--fold', required=False, type=int, help='from 0 to 9', default=0)
    parser.add_argument('--model_name', required=False, type=str, help='ViT, Swin, BEiT', default='ViT')
    parser.add_argument('--pretrained', required=False, type=bool, help='pretrained or not', default=True)
    parser.add_argument('--epoch', type=int, help='training epoch', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)
    parser.add_argument('--save_interval', type=int, help='save after how many epochs', default=1)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.0001)

    return parser.parse_args()


args = arg_parse()

# saving and logging
exp_dir = f'results/{args.model_name}_{args.protocal}{args.fold}'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
data_path = f'./dataset/PHYRE{args.protocal}{args.fold}_{args.batch_size}.pt'
save_path = f'./{exp_dir}/{args.protocal}{args.fold}'

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'{exp_dir}/exp.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

hp = {"model_name": args.model_name, "pretrained": args.pretrained,
      "epoch": args.epoch, "batch_size": args.batch_size, "lr": args.lr,
      "data_path": data_path, "save_path": save_path}
logging.info(hp)

# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# model
if args.model_name == 'ViT':
    print("Using ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=args.pretrained)
    model.head = nn.Linear(768, 2)
elif args.model_name == 'Swin':
    print("Using Swin model...")
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=args.pretrained)
    model.head = nn.Linear(1024, 2)
elif args.model_name == 'BEiT':
    print("Using BEiT model...")
    model = timm.create_model('beit_base_patch16_224', pretrained=args.pretrained)
    model.head = nn.Linear(768, 2)
else:
    raise ValueError(f'The model {args.model_name} is not defined.')

model.to(device)

# optimization
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
loss_fn = torch.nn.BCELoss()

# dataset
if os.path.exists(data_path):
    print(f"Loading train dataset from {data_path}")
    train_loader = torch.load(data_path)
else:
    print(f"Creating PHYRE train dataset of {args.protocal}{args.fold} with a batch size of {args.batch_size}")
    train_set = PHYREO(split='train')
    kwargs = {'pin_memory': True, 'num_workers': 16}
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, **kwargs,
    )

# training
epoch_loss = []
best_loss = 100.
for i in range(args.epoch):
    sum_loss = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        label_one_hot = one_hot(label.to(torch.int64), 2).float().to(device)

        opt.zero_grad()

        out = model(data).squeeze(dim=-1)
        out = nn.Softmax(1)(out)

        pred = torch.argmax(out, dim=-1).float()
        acc = (pred == label).sum() / args.batch_size
        loss = loss_fn(out, label_one_hot)

        loss.backward()
        opt.step()
        scheduler.step()

        sum_loss.append(loss.cpu().detach().numpy())
        print(f'epoch {i} batch {batch_idx} acc: {acc.cpu().detach().numpy():.3f} loss: {loss:.4f}')

    mean_loss = np.mean(sum_loss)
    info = f"#######  epoch {i} : {mean_loss}  #########"
    print(info)
    logging.info(info)

    # save the model parameters after an interval
    if i % args.save_interval == 0:
        torch.save(model.state_dict(), save_path + f'_{i+1}.pt')
        if mean_loss < best_loss:
            best_loss = mean_loss

print(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
logging.info(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")