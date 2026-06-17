import logging
import random
import pickle

import cv2
import h5py
import os
import torch
import numpy as np
from SUREFlow.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


class LiberoDataset():
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 32,
            action_dim: int = 7,
            state_dim: int = 45,
            max_len_data: int = 136,
            chunck_size: int = 1,
            start_idx: int = 0,
            demos_per_task: int = 1,
    ):
        self.data_directory = data_directory
        # Always keep dataset tensors on CPU to avoid CUDA tensors in DataLoader workers
        # self.device = "cpu"
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_len_data = max_len_data
        self.chunck_size = chunck_size
        self.start_idx = start_idx
        self.demos_per_task = demos_per_task

        self.data_dir = sim_framework_path(self.data_directory)
        logging.info("The dataset is loading from {}".format(self.data_dir))  # show the dataset directory

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.data_directory = data_directory

        benchmark_type = os.path.basename(data_directory)
        task_emb_dir = sim_framework_path("language_embeddings")

        with open(task_emb_dir + "/" + benchmark_type + ".pkl", 'rb') as f:
            tasks = pickle.load(f)

        data_embs = []
        actions = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        all_states = []

        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith('.hdf5'):
                continue

            filename = os.path.basename(file).split('.')[0][:-5]
            task_emb = tasks[filename]

            f = h5py.File(os.path.join(self.data_dir, file), 'r')

            log.info("Loading demo: {}".format(file))

            demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])

            # load the states and actions in demos according to demo_keys_list
            for i in indices[start_idx: start_idx + demos_per_task]:

                demo_name = demo_keys_list[i]
                demo = f["data"][demo_name]
                # states_data = demo['states'][:]
                action_data = demo['actions'][:]
                # rewards_data = demo['rewards'][:]
                # dones_data = demo['dones'][:]

                # zero_states = np.zeros((1, self.max_len_data, self.state_dim), dtype=np.float32)
                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                # zero_rewards = np.zeros((1, self.max_len_data), dtype=np.float32)
                # zero_dones = np.zeros((1, self.max_len_data), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                # zero_states[0, :demo_length, :] = states_data  # would be T0, ...,Tn-1, Tn, 0, 0
                # zero_actions[0, :demo_length, :] = action_data
                # zero_rewards[0, :demo_length] = rewards_data
                # zero_dones[0, :demo_length] = dones_data
                # zero_mask[0, :demo_length] = 1

                # the_last_state = states_data[-1][:]
                the_last_action = action_data[-1][:]
                # the_last_reward = rewards_data[-1]
                # the_last_done = dones_data[-1]

                # zero_modelview = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                # zero_inhand = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                model_view = demo['obs']['agentview_rgb'][:]
                eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                joint_states = demo['obs']['joint_states'][:]
                gripper_states = demo['obs']['gripper_states'][:]

                robot_states = np.concatenate((joint_states, gripper_states), axis=-1)

                demo_length = min(
                    demo.attrs["num_samples"],
                    action_data.shape[0],
                    model_view.shape[0],
                    eye_in_hand.shape[0],
                    robot_states.shape[0],
                    self.max_len_data,
                )
                if demo_length < demo.attrs["num_samples"]:
                    log.warning(
                        "Truncating demo %s from %s to %s to fit max_len_data=%s.",
                        demo_name,
                        demo.attrs["num_samples"],
                        demo_length,
                        self.max_len_data,
                    )

                if demo_length > 0:
                    the_last_action = action_data[demo_length - 1][:]

                zero_actions[0, :demo_length, :] = action_data[:demo_length]
                zero_mask[0, :demo_length] = 1

                model_view = model_view[:demo_length]
                eye_in_hand = eye_in_hand[:demo_length]
                robot_states = robot_states[:demo_length]

                # test_img = model_view[0]
                # test_img = test_img[::-1, :, :]
                # test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
                # cv2.imshow("test_img", test_img)
                # cv2.waitKey(0)

                # states.append(zero_states)
                actions.append(zero_actions)
                # rewards.append(zero_rewards)
                # dones.append(zero_dones)
                masks.append(zero_mask)

                agentview_rgb.append(model_view)
                eye_in_hand_rgb.append(eye_in_hand)

                all_states.append(robot_states)

                data_embs.append(task_emb)

            f.close()

        # self.states = torch.from_numpy(np.concatenate(states)).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).float()  # shape: B, T, D

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        self.all_states = all_states

        self.data_embs = data_embs
        self.tasks = tasks

        # self.rewards = torch.from_numpy(np.concatenate(rewards)).float()
        # self.dones = torch.from_numpy(np.concatenate(dones)).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).float()

        self.num_data = len(self.agentview_rgb)

        self.slices = self.get_slices()

    def get_slices(self):  #Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.chunck_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.chunck_size}")
            else:
                slices += [
                    (i, start, start + self.chunck_size) for start in range(T - self.chunck_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = {}

        task_emb = self.data_embs[i]

        agentview_rgb = self.agentview_rgb[i][start:start+1]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start:start+1]

        robot_states = self.all_states[i][start:start+1]

        # Keep on CPU; device transfer handled in training loop
        task_emb = task_emb.float() if isinstance(task_emb, torch.Tensor) else torch.tensor(task_emb, dtype=torch.float32)

        agentview_rgb = torch.from_numpy(agentview_rgb).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        obs["agentview_image"] = agentview_rgb
        obs["eye_in_hand_image"] = eye_in_hand_rgb
        obs["lang_emb"] = task_emb

        obs["robot_states"] = torch.from_numpy(robot_states).float()

        return obs, act, mask
