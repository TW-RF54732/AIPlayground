import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import time

class DiepEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # 基本設定
        self.window_size = 512
        self.player_radius = 10
        self.bullet_radius = 4
        self.bullet_speed = 6
        self.fire_cooldown = 15  # 冷卻時間 (frame 數)
        self.bullet_lifetime = 60

        self.action_space = spaces.MultiBinary(5)  # 上下左右 + 射擊
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_pos = np.array([self.window_size // 2, self.window_size // 2], dtype=np.float32)
        self.bullets = []
        self.cooldown_timer = 0
        self.frame_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # 回傳玩家座標 + 前2顆子彈的位置 + 冷卻狀態
        obs = np.zeros(10, dtype=np.float32)
        obs[0:2] = self.player_pos / self.window_size
        for i, b in enumerate(self.bullets[:2]):
            obs[2 + i * 2: 4 + i * 2] = b["pos"] / self.window_size
        obs[-1] = 1.0 if self.cooldown_timer == 0 else 0.0
        return obs

    def step(self, action):
        dx = dy = 0
        if action[0]: dy -= 3
        if action[1]: dy += 3
        if action[2]: dx -= 3
        if action[3]: dx += 3

        self.player_pos += np.array([dx, dy])
        self.player_pos = np.clip(self.player_pos, self.player_radius, self.window_size - self.player_radius)

        # 處理子彈
        new_bullets = []
        for b in self.bullets:
            b["pos"] += b["dir"] * self.bullet_speed
            b["life"] -= 1
            if 0 <= b["pos"][0] <= self.window_size and 0 <= b["pos"][1] <= self.window_size and b["life"] > 0:
                new_bullets.append(b)
        self.bullets = new_bullets

        if action[4] and self.cooldown_timer == 0:
            self._fire_bullet()

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        self.frame_count += 1

        reward = 0
        terminated = False
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def _fire_bullet(self):
        direction = np.array([1, 0], dtype=np.float32)  # 簡化：子彈往右飛
        bullet = {
            "pos": self.player_pos.copy(),
            "dir": direction,
            "life": self.bullet_lifetime
        }
        self.bullets.append(bullet)
        self.cooldown_timer = self.fire_cooldown

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Simple Diep Gym")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((30, 30, 30))
        pygame.draw.circle(self.window, (0, 255, 0), self.player_pos.astype(int), self.player_radius)
        for b in self.bullets:
            pygame.draw.circle(self.window, (255, 255, 0), b["pos"].astype(int), self.bullet_radius)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

# 測試遊戲
if __name__ == "__main__":
    env = DiepEnv(render_mode="human")
    obs, _ = env.reset()

    env.render()  # ⬅️ 初始化 Pygame 視窗

    running = True
    while running:
        action = [0, 0, 0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: action[0] = 1
        if keys[pygame.K_s]: action[1] = 1
        if keys[pygame.K_a]: action[2] = 1
        if keys[pygame.K_d]: action[3] = 1
        if keys[pygame.K_SPACE]: action[4] = 1

        obs, _, _, _, _ = env.step(action)
        env.render()

    env.close()

