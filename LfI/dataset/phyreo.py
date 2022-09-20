import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from tqdm import tqdm
import phyre

from configs.phyre_cfg import _C as cfg


class PHYREO(Dataset):
    def __init__(self, split):
        self.split = split
        self.input_height, self.input_width = cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH

        protocal = cfg.PHYRE_PROTOCAL
        fold = cfg.PHYRE_FOLD

        num_pos = 1000 if split == 'train' else 100
        num_neg = 1000 if split == 'train' else 100

        eval_setup = f'ball_{protocal}_template'
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold)
        tasks = train_tasks + dev_tasks if split == 'train' else test_tasks
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)

        # all the actions
        cache = phyre.get_default_100k_cache('ball')
        training_data = cache.get_sample(tasks, None)
        # (100000 x 3)
        actions = training_data['actions']
        # (num_tasks x 100000)
        sim_statuses = training_data['simulation_statuses']

        self.simulator = phyre.initialize_simulator(tasks, action_tier)

        self.video_info = np.zeros((0, 4))
        for t_id, t in enumerate(tqdm(tasks)):
            sim_status = sim_statuses[t_id]
            pos_acts = actions[sim_status == 1].copy()
            neg_acts = actions[sim_status == -1].copy()
            np.random.shuffle(pos_acts)
            np.random.shuffle(neg_acts)
            pos_acts = pos_acts[:num_pos]
            neg_acts = neg_acts[:num_neg]
            acts = np.concatenate([pos_acts, neg_acts])
            video_info = np.zeros((acts.shape[0], 4))
            video_info[:, 0] = t_id
            video_info[:, 1:] = acts
            self.video_info = np.concatenate([self.video_info, video_info])

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        task_id, acts = self.video_info[idx, 0], self.video_info[idx, 1:]
        sim = self.simulator.simulate_action(
            int(task_id), acts, stride=60, need_images=True, need_featurized_objects=True
        )
        images = sim.images
        init_image = cv2.resize(images[0], (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

        labels = torch.from_numpy(np.array(int(sim.status == 1), dtype=np.float32))
        data = np.array([phyre.observations_to_float_rgb(init_image)], dtype=np.float).transpose((0, 3, 1, 2))
        data = torch.from_numpy(data.astype(np.float32))

        return data[0], labels


if __name__ == "__main__":
    print(f"Creating PHYRE train dataset of {cfg.PHYRE_PROTOCAL}{cfg.PHYRE_FOLD} with a batch size of {cfg.BATCH_SIZE}")
    train_set = PHYREO(split='train')
    kwargs = {'pin_memory': True, 'num_workers': 16}
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, **kwargs,
    )
    torch.save(train_loader, f'./PHYRE_{cfg.PHYRE_PROTOCAL}{cfg.PHYRE_FOLD}_{cfg.BATCH_SIZE}.pt')
