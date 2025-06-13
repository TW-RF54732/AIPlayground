import pygame
from diep_env import DiepEnv

env = DiepEnv(render_mode="human")
obs, _ = env.reset()

pygame.init()
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    action = None
    if keys[pygame.K_w]: action = 0
    elif keys[pygame.K_s]: action = 1
    elif keys[pygame.K_a]: action = 2
    elif keys[pygame.K_d]: action = 3
    elif pygame.mouse.get_pressed()[0]: action = 4

    if action is not None:
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.render()

env.close()
