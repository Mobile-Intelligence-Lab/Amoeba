import gym
import numpy as np
from gym import spaces
import torch
from dataset import VariableRecordDataset, AgentTrainSet
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminators import SDAE
from utils import action2packet


class AdvEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self,
                 discriminator,
                 state_encoder,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 device,
                 data_penalty=0.5,
                 truncate_penalty=2,
                 time_penalty=0.5,
                 MAX_UNIT=16500,
                 MAX_DELAY=1,
                 max_actions=(0.5, 0.1),
                 max_episode_length=59,
                 enc_dim=256,
                 binary_signal=True,
                 mask_rate=0,
                 action_mode="direction_limited"):
        super().__init__()
        self.discriminator = discriminator
        self.state_encoder = state_encoder
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_generator, self.val_generator = self.get_train_generator(), self.get_val_generator()
        self.test_generator = self.get_test_generator()
        self.device = device
        self.data_penalty = data_penalty
        self.time_penalty = time_penalty
        self.truncate_penalty = truncate_penalty
        self.MAX_UNIT = MAX_UNIT
        self.MAX_DELAY = MAX_DELAY
        self.max_episode_length = max_episode_length
        self.binary_signal = binary_signal
        self.max_actions = max_actions
        self.mask_rate = mask_rate
        self.action_mode = action_mode
        if action_mode == "direction_limited" or action_mode == "addition_only":
            min_actions = [0, 0]
        elif action_mode == "direction_unlimited":
            min_actions = [-1, 0]
        elif action_mode == "pkt_change":
            min_actions = [-1, 0]
        self.action_space = spaces.Box(low=np.array(min_actions), high=np.array(max_actions), shape=(2,))
        self.observation_space = spaces.Box(-1, 1, shape=(enc_dim * 2 + 1 + 2,))

        # per-episode variables
        self.pkt_counter = self.step_c = 0
        self.virtual_adv_flows = []
        self.generated_pkts = []
        self.original_pkts = []
        self.last_gen_hidden_states = None
        self.last_original_hidden_state = None
        self.truncate_num = 0
        self.adv_flow_id = 1
        self.original_flow = None
        self.remaining_bytes = None
        self.last_t = None
        self.mode = False

    def get_train_generator(self):
        X_path = self.train_dataset["X"]
        train_set = DataLoader(AgentTrainSet(X_path, self.MAX_UNIT, self.MAX_DELAY),
                               batch_size=1, shuffle=True, drop_last=True)
        while True:
            for x, _ in tqdm(train_set):
                yield x

    def get_val_generator(self):
        X_path, y_path = self.val_dataset["X"], self.val_dataset["y"]
        val_set = DataLoader(VariableRecordDataset(X_path, y_path, self.MAX_UNIT, self.MAX_DELAY, target=0),
                             batch_size=1, shuffle=False, drop_last=True)
        while True:
            for x, _ in val_set:
                yield x

    def get_test_generator(self):
        X_path, y_path = self.test_dataset["X"], self.test_dataset["y"]
        test_set = DataLoader(VariableRecordDataset(X_path, y_path, self.MAX_UNIT, self.MAX_DELAY, target=0),
                              batch_size=1, shuffle=False, drop_last=True)
        while True:
            for x, _ in test_set:
                yield x

    def get_train_sample(self):
        sample = self.train_generator.__next__()
        sample = sample.view(-1, 2)
        return sample

    def get_val_sample(self):
        sample = self.val_generator.__next__()
        sample = sample.view(-1, 2)
        return sample

    def get_test_sample(self):
        sample = self.test_generator.__next__()
        sample = sample.view(-1, 2)
        return sample

    @staticmethod
    def gen_consecutive_sending_delay():
        return torch.rand(1).item() * 0.002

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action: [altered packet size, altered delay]
        """
        info = {"original_flow": self.original_flow, "completed": 0}
        self.step_c += 1
        terminated = False
        directional_action = action2packet(self.action_mode, self.remaining_bytes, self.last_t, action)

        # gen reward
        self.generated_pkts.append(torch.tensor(directional_action, device=self.device, dtype=torch.float32))
        pkt_size = directional_action[0].item()
        t = directional_action[1].item()
        added_delay = t - self.last_t
        pkt_diff = np.abs(directional_action[0] - self.remaining_bytes)
        reward = self.get_reward(pkt_diff, added_delay)

        # gen next state
        next_pkt = torch.zeros(2)
        if self.remaining_bytes > 0:
            if self.remaining_bytes - pkt_size > 0:
                self.truncate_num += 1
                self.remaining_bytes -= np.maximum(pkt_size, 0)
                next_pkt[0] = self.remaining_bytes
                next_pkt[1] = self.gen_consecutive_sending_delay()
                self.last_t = next_pkt[1].item()
            else:
                self.truncate_num = 0
                self.pkt_counter += 1
                if self.pkt_counter < self.original_flow.size(0):
                    next_pkt = self.original_flow[self.pkt_counter]
                    self.remaining_bytes = next_pkt[0].item()
                    self.last_t = next_pkt[1].item()
                else:
                    self.virtual_adv_flows.append(torch.stack(self.generated_pkts, dim=0))
                    terminated = True
                    info['completed'] = 1
        else:
            if self.remaining_bytes - pkt_size < 0:
                self.truncate_num += 1
                self.remaining_bytes -= np.minimum(pkt_size, 0)
                next_pkt[0] = self.remaining_bytes
                next_pkt[1] = self.gen_consecutive_sending_delay()
                self.last_t = next_pkt[1].item()
            else:
                self.truncate_num = 0
                self.pkt_counter += 1
                if self.pkt_counter < self.original_flow.size(0):
                    next_pkt = self.original_flow[self.pkt_counter]
                    self.remaining_bytes = next_pkt[0].item()
                    self.last_t = next_pkt[1].item()
                else:
                    self.virtual_adv_flows.append(torch.stack(self.generated_pkts, dim=0))
                    terminated = True
                    info["completed"] = 1

        # reach the capacity of state encoder
        if self.step_c == self.max_episode_length:
            self.adv_flow_id += 1
            self.virtual_adv_flows.append(torch.stack(self.generated_pkts, dim=0))
            if self.mode == "train" or self.adv_flow_id > 3:
                # allow truncating a flow up to 3 times (adjustable)
                terminated = True
            else:
                self.step_c = 0
                self.reset_virtual_flow_states()

        next_pkt = next_pkt.to(self.device)
        self.original_pkts.append(next_pkt)
        next_state = self.get_encoded_states()

        info["virtual_adv_flows"] = self.virtual_adv_flows
        return next_state, reward.item(), terminated, info

    def get_encoded_states(self):
        with torch.no_grad():
            if self.last_original_hidden_state is None:
                encoded_original_pkts, self.last_original_hidden_state = self.state_encoder.step(
                    self.original_pkts[-1].view(1, 1, 2))
            else:
                encoded_original_pkts, self.last_original_hidden_state = self.state_encoder.step(
                    self.original_pkts[-1].view(1, 1, 2), self.last_original_hidden_state)
            if self.last_gen_hidden_states is None:
                if len(self.generated_pkts) == 0:
                    # no adv packets generated yet
                    encoded_adv_pkts = torch.zeros_like(encoded_original_pkts, device=self.device)
                else:
                    encoded_adv_pkts, self.last_gen_hidden_states = self.state_encoder.step(
                        self.generated_pkts[-1].view(1, 1, 2))
            else:
                encoded_adv_pkts, self.last_gen_hidden_states = self.state_encoder.step(
                    self.generated_pkts[-1].view(1, 1, 2), self.last_gen_hidden_states)

            encoded_original_pkts = encoded_original_pkts.cpu().numpy().reshape((-1,))
            encoded_adv_pkts = encoded_adv_pkts.cpu().numpy().reshape((-1,))
            next_pkt = self.original_pkts[-1].cpu().numpy()
            binary_signal = np.ones((1,)) if self.truncate_num != 0 else np.zeros((1,))
            next_encoded_state = np.concatenate([encoded_original_pkts, encoded_adv_pkts, binary_signal, next_pkt],
                                                axis=0)
        return next_encoded_state

    def get_reward(self, pkt_diff, added_delay):
        # compute reward
        with torch.no_grad():
            generated_flow = torch.stack(self.generated_pkts, dim=0).view(1, -1, 2)
            if isinstance(self.discriminator, SDAE):
                clf_result = self.discriminator(generated_flow)[0]
            else:
                clf_result = self.discriminator(generated_flow)
        if self.binary_signal:
            if clf_result > 0.5:
                visible_result = torch.ones((1,))
            else:
                visible_result = torch.zeros((1,))
            if np.random.uniform() < self.mask_rate:
                # unclear feedback
                visible_result = torch.ones((1,)) * 0.5
            reward = visible_result.to(
                self.device) - self.data_penalty * pkt_diff - self.time_penalty * added_delay - self.truncate_num * self.truncate_penalty

        else:
            reward = clf_result - self.data_penalty * \
                     pkt_diff - self.time_penalty * added_delay - self.truncate_num * self.truncate_penalty

        return reward

    def reset_virtual_flow_states(self):
        self.generated_pkts = []
        self.original_pkts = []
        self.last_gen_hidden_states = None
        self.last_original_hidden_state = None
        self.truncate_num = 0

    def reset(self, mode="train"):
        """
        Important: the observation must be a numpy array
        :return: (np.array), format:
        [observed_state, generated_state, observation_p, observation_t]
        """
        # reset variables
        self.mode = mode
        self.pkt_counter = self.step_c = 0
        self.virtual_adv_flows = []
        self.reset_virtual_flow_states()
        self.adv_flow_id = 1
        if self.mode == "train":
            self.original_flow = self.get_train_sample()
        elif self.mode == "val":
            self.original_flow = self.get_val_sample()
        else:
            self.original_flow = self.get_test_sample()

        # encode state
        init_ob = self.original_flow[self.pkt_counter]
        init_ob = init_ob.to(self.device)
        self.original_pkts.append(init_ob)
        self.remaining_bytes = init_ob[0].item()
        self.last_t = init_ob[1].item()
        state = self.get_encoded_states()
        return state

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
