import sys
import pygame
import numpy as np
import cupy as cp
from src.models.mlp import MLP

# model
model = MLP.load_model("model/model.npz", "cross_entropy")

# pygame stuff
pygame.init()
clock = pygame.time.Clock()
WIDTH, HEIGHT = 280, 280
FPS = 60
radius = 8

black = "#000000"
white = "#ffffff"

screen = pygame.display.set_mode((WIDTH, HEIGHT))

# brush to draw
brush = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
pygame.draw.circle(brush, white, (radius, radius), radius)

canvas = pygame.Surface((280, 280))
canvas.fill(black)

img_preview = pygame.Surface((28, 28))


def linear_interpol(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    d = np.hypot(x2 - x1, y2 - y1)
    path = []
    for i in range(int(d)):
        xi = x1 + (x2 - x1) * i / d
        yi = y1 + (y2 - y1) * i / d
        path.append((xi, yi))

    return path


last_pos = None
scaled_canvas = None
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
            sys.exit()
        if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
            canvas.fill(black)

    x, y = pygame.mouse.get_pos()
    curr_poss = (x, y)
    mb = pygame.mouse.get_pressed()
    if mb[0]:
        if last_pos:
            path = linear_interpol(last_pos, curr_poss)
            for p in path:
                canvas.blit(brush, (p[0] - radius, p[1] - radius))
        else:
            canvas.blit(brush, (x - radius, y - radius))

        scaled_canvas = pygame.transform.smoothscale(canvas, (28, 28))
        img_array = pygame.surfarray.array3d(scaled_canvas)
        img_array = np.transpose(img_array, (1, 0, 2))
        img_gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        img_gray /= 255
        img_flat = img_gray.reshape(1, 28 * 28)
        pred_vec = model.predict(cp.asarray(img_flat))
        pred = cp.argmax(pred_vec)
        print("--" * 20)
        print(f"Predicted value: {pred}")
        print("--" * 20)

        last_pos = curr_poss
    else:
        last_pos = None

    screen.fill(white)
    if scaled_canvas:
        screen.blit(scaled_canvas, (300, 300))
    screen.blit(canvas, (0, 0))
    pygame.display.update()
    clock.tick(FPS)
