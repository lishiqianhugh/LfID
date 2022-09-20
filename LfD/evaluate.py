import cv2
import os
import torch
import phyre
import random
import numpy as np
from tqdm import tqdm
import pdb
from torch import nn
from Network.timesformer.models.vit import TimeSformer
from Network.predrnn.core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2
import argparse
from configs.phyre_cfg import _C as cfg
from utils.predrnn_util import reshape_patch,reshape_patch_back
import logging


def arg_parse():
    parser = argparse.ArgumentParser(description='VIT_PHYRE Parameters')
    parser.add_argument('--protocal', type=str,required=True, help='within or cross', default='within')
    parser.add_argument('--fold', type=int, help='from 0 to 9', default=0)
    parser.add_argument('--epoch', type=int, required=True,help='epoch', default=10)
    parser.add_argument('--batch_size', type=int, help='test batch size', default=256)
    parser.add_argument('--data_mode', required=True, type=str, help='e2e or pap', default='e2e')
    parser.add_argument('--vit_path', type=str, help='saved vit model', default='Network/VIT_Params/within0/within0_128_10.pt')
    parser.add_argument('--TSF_path', type=str, help='saved TSF model', default='Network/TimeSformer_Params/TSF_dynamic_within/within0_40_10.pt')
    parser.add_argument('--model_name', required=False, type=str, help='use which model', default='TimeSFormer')
    parser.add_argument('--pred_model', required=False, type=str, help='RPIN or ConvLSTM or action_cond_predrnn_v2', default='predrnn_v2')
    parser.add_argument('--dataset', required=False, type=str, help='using PHYRE or SS', default='PHYRE')
    parser.add_argument('--total_length', type=int, help='the gt and the prediction frames number', default=cfg.INPUT.INPUT_SIZE + cfg.NET.PRED_SIZE)
    parser.add_argument('--input_length', type=int, default=cfg.INPUT.INPUT_SIZE)
    parser.add_argument('--patch_size', type=int, help='patch size for predRNN', default=8)
    parser.add_argument('--img_width', type=int, help='input size for prediciton model', default=224)
    parser.add_argument('--device', type=str, help='device', default=device)
    parser.add_argument('--pred_model_path', type=str, default='results/total_3e2e/within0_16_pred_model9.pt')
    parser.add_argument('--model_path', type=str, default='results/total_3e2e/within0_16_9.pt')
    return parser.parse_args()

# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
args = arg_parse()
# log
exp_dir = f'auccess/{args.data_mode}_{args.protocal}_{args.epoch}'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'{exp_dir}/exp.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


hp = {'data_mode':args.data_mode,'protocal':args.protocal,'epoch': args.epoch,'pred_model_path':args.pred_model_path,'model_path':args.model_path}
logging.info(hp)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


print(f"Resuming prediction model...")
if args.pred_model == 'predrnn_v2': #FIXME:
    num_hidden = '224,224,224,224'
    num_hidden = [int(x) for x in num_hidden.split(',')]
    num_layers = len(num_hidden)
    pred_model = eval(args.pred_model + '.RNN')(num_layers,num_hidden,args)
    pred_model.to(device)
    pred_model.load_state_dict(torch.load(f'{args.pred_model_path}'))

print(f"Resuming pretrained TSF model...")
#TODO:
model = TimeSformer(img_size=224, num_classes=2, num_frames=8, attention_type='divided_space_time',
                    pretrained_model='Network/TimeSformer_pretrained/TimeSformer_divST_8x32_224_K600.pyth')

model.load_state_dict(torch.load(f'{args.model_path}'))
model.to(device)


#prepare for evaluation
start_id, end_id = 0, 25
random.seed(0)
np.random.seed(0)
# protocal, fold_id = C.PHYRE_PROTOCAL, C.PHYRE_FOLD
protocal, fold_id = args.protocal, args.fold
print(f'testing using protocal {protocal} and fold {fold_id}')

# setup the PHYRE evaluation split
eval_setup = f'ball_{protocal}_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
_, _, test_tasks = phyre.get_fold(eval_setup, fold_id)  # PHYRE setup
candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]  # filter tasks
test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
simulator = phyre.initialize_simulator(test_list, action_tier)

# the action candidates are provided by the author of PHYRE benchmark
num_actions = 10000
cache = phyre.get_default_100k_cache('ball')
acts = cache.action_array[:num_actions]
training_data = cache.get_sample(test_list, None)

# some statistics variable when doing the evaluation
auccess = np.zeros((len(test_list), 100))
# batched_pred = C.SOLVER.BATCH_SIZE * 10
batched_pred = args.batch_size
objs_color = None

with torch.no_grad():
    t_list = tqdm(test_list, 'Task')
    all_data, all_rois,all_valid,all_real_input_flag,all_ini= [], [], [], [], []
    for task_id, task in enumerate(t_list):
        sim_statuses = training_data['simulation_statuses'][task_id]
        confs, successes = [], []

        # act_list = tqdm(acts, 'Candidate Action', leave=False)
        for act_id, act in enumerate(acts):
            sim = simulator.simulate_action(task_id, act, stride=60, need_images=True, need_featurized_objects=True)
            # assert sim.status == sim_statuses[act_id], 'sanity check not passed'
            # TODO:
            if sim.status == phyre.SimulationStatus.INVALID_INPUT:
                if act_id == len(acts) - 1 and len(all_data) > 0:
                      # final action is invalid
                    future_imgs,_ = pred_model(torch.cat(all_data).to(device),torch.cat(all_real_input_flag).to(device))
                    future_imgs = reshape_patch_back(future_imgs, args.patch_size)
                    inis = torch.cat(all_ini).to(device)
                    pred_imgs = torch.cat((inis, future_imgs),dim=1).permute((0,4,1,2,3))
                    out = model(pred_imgs).squeeze(dim=-1)
                    out = nn.Softmax(1)(out)
                    conf_t = out[:, 1]
                    confs = confs + conf_t.cpu().detach().numpy().tolist()
                    all_data = []
                    all_real_input_flag = []
                    all_ini = []
                continue
            successes.append(sim.status == phyre.SimulationStatus.SOLVED)
            images = sim.images
           
            tmps = []
            if args.data_mode == 'intuition':
                for i in range(cfg.INPUT.INPUT_SIZE):
                    tmp = cv2.resize(images[0], (cfg.INPUT.INPUT_WIDTH, cfg.INPUT.INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    tmps.append(tmp)
        
            else:
                for i in range(args.total_length):
                    try:
                        tmp = cv2.resize(images[i], (cfg.INPUT.INPUT_WIDTH, cfg.INPUT.INPUT_HEIGHT),
                                        interpolation=cv2.INTER_NEAREST)
                        tmps.append(tmp)
                    except:
                        pdb.set_trace()
                        for j in range(cfg.INPUT.INPUT_SIZE - i):
                            tmp = cv2.resize(images[i - 1], (cfg.INPUT.INPUT_WIDTH, cfg.INPUT.INPUT_HEIGHT),
                                            interpolation=cv2.INTER_NEAREST)
                            tmps.append(tmp)
                        break
            
            data = np.array([phyre.observations_to_float_rgb(image) for image in tmps],
                            dtype=np.float).transpose((0, 3, 1, 2))[None,:]

            data = torch.from_numpy(data.astype(np.float32))  # torch.Size([1, 3, T, 224, 224])
            if args.pred_model == 'predrnn_v2':
                data = data.permute(0,1,3,4,2)
                ini = data[:,0,:].unsqueeze(1)
                data_in = reshape_patch(data, args.patch_size)
                real_input_flag = torch.zeros(
                (1,
                args.total_length - 2,                  #把需要预测的mask掉了
                args.img_width // args.patch_size,
                args.img_width // args.patch_size,
                args.patch_size ** 2 * 3))
                real_input_flag[:, :args.input_length - 1, :, :] = 1.0   
            all_data.append(data_in)
            all_real_input_flag.append(real_input_flag)
            all_ini.append(ini)


            if len(all_data) % batched_pred == 0 or act_id == len(acts) - 1:
                future_imgs,_ = pred_model(torch.cat(all_data).to(device),torch.cat(all_real_input_flag).to(device))
                future_imgs = reshape_patch_back(future_imgs, args.patch_size)
                inis = torch.cat(all_ini).to(device)
                pred_imgs = torch.cat((inis, future_imgs),dim=1).permute((0,4,1,2,3))
                out = model(pred_imgs).squeeze(dim=-1)
                out = nn.Softmax(1)(out)
                conf_t = out[:, 1]
                confs = confs + conf_t.cpu().detach().numpy().tolist()
                all_data = []
                all_real_input_flag = []
                all_ini = []

        info = f'current AUCCESS: '
        top_acc = np.array(successes)[np.argsort(confs)[::-1]]
        for i in range(100):
            auccess[task_id, i] = int(np.sum(top_acc[:i + 1]) > 0)
            # 0/1 represents for whether task_id has successed with i attempts
        w = np.array([np.log(k + 1) - np.log(k) for k in range(1, 101)])
        s = auccess[:task_id + 1].sum(0) / auccess[:task_id + 1].shape[0]
        info += f'{np.sum(w * s) / np.sum(w) * 100:.2f}'
        t_list.set_description(info)
        logging.info(info)
