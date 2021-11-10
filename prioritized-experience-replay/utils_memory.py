from typing import (
    Tuple,
)
import torch
import random
from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

#训练时存储前面的学习经验，重复利用数据
class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 100001
        self.__m_states = torch.zeros(
            (2*capacity+1, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((2*capacity+1, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((2*capacity+1, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((2*capacity+1, 1), dtype=torch.bool)
        self.__m_error = torch.zeros((2*capacity+1, 1), dtype=torch.float)
    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            values: float,
            expected: float
    ) -> None:
        diff = torch.zeros((1,1),dtype=torch.float)
        diff[0][0] = abs(expected-values)
        if self.__size == self.__capacity:
            pre = self.__m_error[self.__pos, 0]
            pre_pos = self.__pos
            while pre_pos != 0:
                self.__m_error[pre_pos, 0] -= pre
                pre_pos = pre_pos//2
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        pre_pos = self.__pos
        while pre_pos != 0:
            self.__m_error[pre_pos,0] += diff[0][0]
            pre_pos = pre_pos // 2
        self.__pos = 100001 if self.__pos == 2*self.__capacity else self.__pos+1
        self.__size = min(self.__capacity, self.__size+1)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        dice = [random.uniform(0, self.__m_error[1, 0]) for i in range(batch_size)]
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        num = 0
        for i in dice:
            pos = 1
            while pos <= 100000:
                if i <= self.__m_error[2 * pos, 0]:
                    pos = 2*pos
                else:
                    i -= self.__m_error[2*pos, 0]
                    pos = 2*pos+1
            if(pos>200000):
                pos = 200000
            indices[num] = pos
            num += 1
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size
