import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage.clear()
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)  # 小括号代表元组Tuple
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # idxes表示随机选择的样本的下标
    def _encode_sample(self, idxes, agent_idx):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.concatenate(obs_t[:]))  # 旧状态
            actions.append(action)
            rewards.append(reward[agent_idx])
            obses_tp1.append(np.concatenate(obs_tp1[:]))  # 新状态
            dones.append(done[agent_idx])
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    # 随机选择
    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage)-1) for _ in range(batch_size)]

    # 不再是随机选择，而是选择最新填入的样本
    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, agent_idx):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes, agent_idx)

if __name__ == '__main__':
    pass