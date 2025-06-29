# cartpole_dqn_project_v2.py – 改进版 DQN（增强准确率 + 动态步数叠加）
# -----------------------------------------------------------------------------
# 改进内容：
#   1. 强化网络结构：增加隐藏层
#   2. 使用 Huber Loss 替代 MSE
#   3. 软更新目标网络（Soft Target Update）
#   4. reward clipping
# -----------------------------------------------------------------------------
import os, csv, random, imageio
from collections import deque
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from PIL import Image, ImageDraw, ImageFont

SEED                 = 42
GAMMA                = 0.99
LR                   = 5e-4
BATCH_SIZE           = 64
BUFFER_SIZE          = 100_000
EPS_START, EPS_END   = 1.0, 0.01
EPS_DECAY            = 0.99
MAX_EPISODES         = 2000
EARLY_STOP_REWARD    = 195
EARLY_STOP_WINDOW    = 100
BEST_WINDOW          = 50
TAU                  = 0.005

BEST_MODEL_FN        = "best_dqn_cartpole.pth"
LAST_MODEL_FN        = "dqn_cartpole_last.pth"
LOG_CSV              = "reward_log.csv"
GIF_NAME             = "cartpole_demo.gif"

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        r = max(min(r, 1.0), -1.0)  # reward clipping
        self.buffer.append((np.asarray(s, np.float32), a, r, np.asarray(s2, np.float32), d))
    def sample(self, batch):
        s, a, r, s2, d = zip(*random.sample(self.buffer, batch))
        return (
            torch.as_tensor(np.vstack(s)),
            torch.as_tensor(a, dtype=torch.long),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(np.vstack(s2)),
            torch.as_tensor(d, dtype=torch.float32),
        )
    def __len__(self):
        return len(self.buffer)

def _get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    candidates: List[str] = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT = _get_font(18)

def train():
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt = optim.Adam(policy_net.parameters(), lr=LR)
    buf = ReplayBuffer(BUFFER_SIZE)

    eps = EPS_START
    rewards = []
    best_ma = -np.inf

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["Episode", "Reward", "Epsilon"])

    for ep in range(MAX_EPISODES):
        state, _ = env.reset()
        done, total = False, 0

        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(policy_net(torch.as_tensor(state).unsqueeze(0)).argmax())
            nxt, r, term, trunc, _ = env.step(action)
            done = term or trunc
            buf.push(state, action, r, nxt, done)
            state = nxt
            total += r

            if len(buf) >= BATCH_SIZE:
                s, a, r_, s2, d_ = buf.sample(BATCH_SIZE)
                q_sa = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                    target = r_ + GAMMA * q_next * (1 - d_)
                loss = nn.functional.smooth_l1_loss(q_sa, target)
                opt.zero_grad(); loss.backward(); opt.step()

                for tp, lp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.copy_(TAU * lp.data + (1 - TAU) * tp.data)

        rewards.append(total)
        eps = max(EPS_END, eps * EPS_DECAY)
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, total, round(eps, 3)])

        if len(rewards) >= BEST_WINDOW:
            cur_ma = np.mean(rewards[-BEST_WINDOW:])
            if cur_ma > best_ma:
                best_ma = cur_ma
                torch.save(policy_net.state_dict(), BEST_MODEL_FN)
        if len(rewards) >= EARLY_STOP_WINDOW and np.mean(rewards[-EARLY_STOP_WINDOW:]) >= EARLY_STOP_REWARD:
            print(f"Early stopping at episode {ep+1}, avg reward={np.mean(rewards[-EARLY_STOP_WINDOW:]):.1f}")
            break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:3d} | Reward {total:4.0f} | ε={eps:.3f} | Best MA(50)={best_ma:.1f}")

    env.close()
    torch.save(policy_net.state_dict(), LAST_MODEL_FN)

    plt.plot(rewards, label="Episode Reward", alpha=0.4)
    if len(rewards) >= 10:
        ma10 = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), ma10, label="MA(10)", linewidth=2)
    plt.axhline(195, color="red", linestyle="--", label="Solved=195")
    plt.title("DQN on CartPole‑v1"); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.grid(); plt.legend(); plt.tight_layout(); plt.savefig("reward_curve.png"); plt.show()

def evaluate(model_path: str = BEST_MODEL_FN, episodes: int = 100, record_gif: bool = True, gif_name: str = GIF_NAME):
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    env = gym.make("CartPole-v1", render_mode="rgb_array" if record_gif else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = QNetwork(state_dim, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    totals, frames = [], []
    for ep in range(episodes):
        s, _ = env.reset()
        done, tot, step = False, 0, 0
        while not done:
            if record_gif and ep == 0:
                arr = env.render()
                img = Image.fromarray(arr)
                draw = ImageDraw.Draw(img)
                draw.rectangle((5, 5, 140, 30), fill=(0, 0, 0))
                draw.text((10, 8), f"Step: {step}", fill=(255, 255, 255), font=FONT)
                frames.append(np.array(img))
            with torch.no_grad():
                a = int(policy(torch.as_tensor(s).unsqueeze(0)).argmax())
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            tot += r; step += 1
        totals.append(tot)

    env.close()
    print(f"\n>>> Evaluation: avg reward = {np.mean(totals):.1f} ± {np.std(totals):.1f}")
    print(f"    Success rate (≥195) = {np.mean(np.array(totals) >= 195) * 100:.1f}% over {episodes} runs\n")

    if record_gif and frames:
        imageio.mimsave(gif_name, frames, fps=30)
        print(f"Demo GIF saved ⇒ {gif_name}\n")

if __name__ == "__main__":
    train()
    evaluate()
