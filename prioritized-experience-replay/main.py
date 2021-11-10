from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
os.mkdir(SAVE_PREFIX)

#前面这一块应该是环境的搭建，用库创建atari游戏
torch.manual_seed(new_seed())
device = torch.device("cuda")
env = MyEnv(device)
"""agent代表玩家本身。初始化agent的参数，agent包含动作选择、策略学习的方法。"""
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    restore = "model_010"
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

"""
iterable  : iterable, optional
            Iterable to decorate with a progressbar.
            Leave blank to manually manage the updates.
total  : int or float, optional
            The number of expected iterations. If unspecified,
            len(iterable) is used if possible. If float("inf") or as a last
            resort, only basic progress statistics are displayed
            (no ETA, no progressbar).
            If `gui` is True and this parameter needs subsequent updating,
            specify an initial arbitrary large positive number
ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes the progressbar to stay within this bound.
            If unspecified, attempts to use environment width. The
            fallback is a meter width of 10 and no limit for the counter and
            statistics. If 0, will not print any meter (only stats).
leave  : bool, optional
            If [default: True], keeps all traces of the progressbar
            upon termination of iteration.
            If `None`, will leave only if `position` is `0`.
unit  : str, optional
            String that will be used to define the unit of each iteration
            [default: it].
"""
progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        """reset resets and initializes the underlying gym environment.
        maybe observation represent the environment"""
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    """run suggests an action for the given state.
    if memory is bigger than warm_steps then reduce eps"""
    action = agent.run(state, training)
    """step forwards an action to the environment and returns the newest
            observation, the reward, and an bool value indicating whether the
            episode is terminated."""
    """根据当前地图情况获得下一步的obs"""
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    """make_folded_state将队列列表化并变成1*n大小的二维数组，这样一整个状态可以作为一个成员存储在memory"""
    """暂时推测用的是维度是4的神经网络"""
    with torch.no_grad():
        folded_state = env.make_folded_state(obs_queue)
        indices = [0 for i in range(32)]
        state = folded_state[indices, :4].to(device).float()
        state_next = folded_state[indices, 1:].to(device).float()
        pre_action = torch.zeros((1, 1), dtype=torch.long)
        pre_action[0][0] = action
        values = agent.policy(state.to(device).float()).gather(1, pre_action[indices].to(device))
        values_next = agent.target(state_next.to(device).float()).max(1).values.detach()
        expected = (agent.gamma * values_next.unsqueeze(1)) * \
               (1. - done) + reward
        values = sum(values[0])
        expected = sum(expected[0])
        memory.push(folded_state, action, reward, done,values,expected)

    if step % POLICY_UPDATE == 0 and training:
        """learn trains the value network via TD-learning."""
        """从memory中取出一个样本进行学习"""
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()

    #每过EVALUATE_FREQ次进行一次保存
    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
