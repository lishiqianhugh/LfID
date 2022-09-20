import os
import cv2
import torch
import phyre
import pickle
import random
import numpy as np
from tqdm import tqdm
import timm
from torch import nn
import matplotlib.pyplot as plt

import argparse
from utils.config import _C as cfg


def arg_parse():
    parser = argparse.ArgumentParser(description='VIT_PHYRE Parameters')
    parser.add_argument('--model', required=False, help='path to tset model', type=str, default='within0_128_10.pt')
    parser.add_argument('--protocal', type=str, help='within or cross', default='within')
    parser.add_argument('--fold', type=int, help='from 0 to 9', default=0)
    parser.add_argument('--batch_size', type=int, help='test batch size', default=128)

    return parser.parse_args()


args = arg_parse()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("Loading model...")
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(768, 2)
model.load_state_dict(torch.load(f"../exp_within_0_10_128_0.0001/{args.model}"))
model.to(device)


for start_id in tqdm(range(25)):
    # start_id = 0
    end_id = start_id + 1
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
    ti = 0
    test_list = test_list[ti:ti+1]
    simulator = phyre.initialize_simulator(test_list, action_tier)
    assert len(test_list) == 1

    # the action candidates are provided by the author of PHYRE benchmark
    num_actions = 10000
    cache = phyre.get_default_100k_cache('ball')
    acts = cache.action_array[:num_actions]
    training_data = cache.get_sample(test_list, None)

    batched_pred = args.batch_size

    all_data = []

    with torch.no_grad():
        for template_id, task in enumerate(test_list):
            sim_statuses = training_data['simulation_statuses'][template_id]
            confs, successes = [], []
            valid_acts = []

            # act_list = tqdm(acts, 'Candidate Action')
            for act_id, act in enumerate(acts):
                sim = simulator.simulate_action(template_id, act, stride=60, need_images=True, need_featurized_objects=True)
                # assert sim.status == sim_statuses[act_id], 'sanity check not passed'
                if sim.status == phyre.SimulationStatus.INVALID_INPUT:
                    if act_id == len(acts) - 1 and len(all_data) > 0:  # final action is invalid
                        # raise ValueError("invalid act")
                        out = model(torch.cat(all_data).to(device))
                        out = nn.Softmax(1)(out)
                        conf_t = out[:, 1]
                        confs = confs + conf_t.cpu().detach().numpy().tolist()
                        all_data = []
                    continue
                valid_acts.append(act)
                successes.append(sim.status == phyre.SimulationStatus.SOLVED)
                image = cv2.resize(sim.images[0], (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT),
                                   interpolation=cv2.INTER_NEAREST)

                image = phyre.observations_to_float_rgb(image)
                data = image.transpose((2, 0, 1))[None, None, :]
                data = torch.from_numpy(data.astype(np.float32))  # torch.Size([1, 1, 3, 224, 224])
                all_data.append(data[0])

                if len(all_data) % batched_pred == 0 or act_id == len(acts) - 1:
                    out = model(torch.cat(all_data).to(device))
                    out = nn.Softmax(1)(out)
                    conf_t = out[:, 1]
                    confs = confs + conf_t.cpu().detach().numpy().tolist()
                    all_data = []

            # save acts and confs
    confs = np.array(confs, dtype=float)[:, None]
    valid_acts = np.array(valid_acts, dtype=float)
    distr_map = np.concatenate((valid_acts, confs), axis=-1)
    save_path = f'./data_map/map_{start_id}_{ti}.hkl'
    with open(save_path, 'wb') as f:
        pickle.dump(distr_map, f)



