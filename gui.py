"""
Real-time MNIST Digit Recognition.

Draw a digit on the canvas using the mouse. The model will predict
the digit in real-time and show probabilities on the panel.

Controls:
    - Left Click: Draw
    - 'C' Key: Clear Canvas
    - 'Q' Key: Quit
"""

import os
import sys
import pygame
import cupy as cp
import numpy as np
from src.models.mlp import MLP

path = "model/model.npz"

if not os.path.exists(path):
    raise RuntimeError(
        "Could not load the model. Make sure to train the model and save it first"
    )

model = MLP.load_model(path, "cross_entropy")

# UI constants
WIDTH, HEIGHT = 600, 400
CANVAS_PANEL_WIDTH = 350
PROB_BAR_HEIGHT = 14
PROB_BAR_WIDTH = 120
BRUSH_RADIUS = 8

# Catppuccin-inspired color theme
BLACK = "#000000"
WHITE = "#ffffff"
BG_COLOR = "#1e1e2e"
PANEL_COLOR = "#181825"
CANVAS_BG = "#11111b"
ACCENT = "#89b4fa"
ACCENT_STRONG = "#74c7ec"
TEXT_COLOR = "#cdd6f4"
MUTED = "#6c7086"
BLUE = "#4f90b3"

FPS = 60

# Pygame setup
pygame.init()
pygame.display.set_caption("MNIST Digit Recognition")

screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Circular brush for drawing
brush = pygame.Surface((BRUSH_RADIUS * 2, BRUSH_RADIUS * 2), pygame.SRCALPHA)
pygame.draw.circle(brush, WHITE, (BRUSH_RADIUS, BRUSH_RADIUS), BRUSH_RADIUS)

canvas = pygame.Surface((280, 280))
canvas.fill(BLACK)

# Fonts
font_main = pygame.font.SysFont("Segoe UI", 24)
font_bold = pygame.font.SysFont("Segoe UI", 42, bold=True)
font_small = pygame.font.SysFont("Segoe UI", 18)

clock = pygame.time.Clock()


def linear_interpolate(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple]:
    """
    Calculates points between two coordinates for smooth line drawing.

    Args:
        p1 (tuple): Start (x, y) coordinates.
        p2 (tuple): End (x, y) coordinates.

    Returns:
        list[tuple]: A list of intermediate (x, y) points.
    """
    x1, y1 = p1
    x2, y2 = p2

    d = np.hypot(x2 - x1, y2 - y1)
    path = []
    for i in range(int(d)):
        xi = x1 + (x2 - x1) * i / d
        yi = y1 + (y2 - y1) * i / d
        path.append((xi, yi))

    return path


def draw_prob_bar(
    surf: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    color: str | tuple,
    label: str,
) -> None:
    """
    Renders a progress bar representing prediction probability.

    Args:
        surf (pygame.Surface): The surface to draw on.
        x (int): X-coordinate of the bar.
        y (int): Y-coordinate of the bar.
        width (int): Total width of the bar.
        height (int): Height of the bar.
        progress (float): Completion value between 0.0 and 1.0.
        color (str/tuple): Hex or RGB color for the filled part.
        label (str): Text label (digit) to display next to the bar.

    Returns:
        None
    """
    pygame.draw.rect(surf, CANVAS_BG, (x, y, width, height), border_radius=10)

    bar_width = int(width * progress)
    if bar_width > 0:
        pygame.draw.rect(surf, color, (x, y, bar_width, height), border_radius=10)

    txt = font_main.render(label, True, TEXT_COLOR)
    surf.blit(txt, (x - 25, y - 5))


def process_img(canvas: pygame.Surface) -> tuple[pygame.Surface, np.ndarray, int]:
    """
    Processes the Pygame canvas into a format suitable for the MLP model.

    Scales the 280x280 canvas to 28x28, converts to grayscale,
    normalizes pixel values, and runs the prediction.

    Args:
        canvas (pygame.Surface): The raw drawing canvas.

    Returns:
        tuple: (scaled_surface, probability_array, predicted_digit)
    """
    # Resize to 28x28 (MNIST input size)
    scaled_canvas = pygame.transform.smoothscale(canvas, (28, 28))
    img_array = pygame.surfarray.array3d(scaled_canvas)

    # Pygame uses (x, y, channel) format which must be turned into numpy (y, x, channel)
    # swapping rows with columns
    img_array = np.transpose(img_array, (1, 0, 2))

    # Converting image to black and white using BT.601 standard
    img_gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

    # Normalization and flattening the data to the format which MLP was trained on
    img_gray /= 255
    img_flat = img_gray.reshape(1, 28 * 28)

    pred_vec = model.predict(cp.asarray(img_flat))
    pred = int(cp.argmax(pred_vec))
    probs = pred_vec.get().flatten()
    return scaled_canvas, probs, pred


# UI components to draw in the main loop
canvas_rect = canvas.get_rect(center=(CANVAS_PANEL_WIDTH // 2, HEIGHT // 2))
left_txt = font_small.render("C: Clear | Q: Quit", True, MUTED)
# Box for predicted digit
result_bg = pygame.Rect(CANVAS_PANEL_WIDTH + 50, 10, 60, 60)
# Position for the scaled down image version
preview_pos = (CANVAS_PANEL_WIDTH + 150, 25)

last_pos = None
probs = np.zeros(10)
pred = None
scaled_canvas = pygame.Surface((28, 28))
scaled_canvas.fill(BLACK)

# Main loop
while True:
    # Event handling
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
            sys.exit()
        if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
            canvas.fill(BLACK)
            probs = np.zeros(10)
            pred = None
            scaled_canvas.fill(BLACK)

    screen.fill(BG_COLOR)

    # Drawing logic
    x, y = pygame.mouse.get_pos()

    # Adjust coordinates to the canvas position
    x -= (CANVAS_PANEL_WIDTH - canvas.get_width()) // 2
    y -= (HEIGHT - canvas.get_height()) // 2
    curr_pos = (x, y)
    mb = pygame.mouse.get_pressed()
    if mb[0]:
        if last_pos:
            path = linear_interpolate(last_pos, curr_pos)
            for p in path:
                canvas.blit(brush, (p[0] - BRUSH_RADIUS, p[1] - BRUSH_RADIUS))
        else:
            canvas.blit(brush, (x - BRUSH_RADIUS, y - BRUSH_RADIUS))

        last_pos = curr_pos
        scaled_canvas, probs, pred = process_img(canvas)
    else:
        last_pos = None

    # UI RENDERING
    # Left panel (drawing area)
    pygame.draw.rect(
        screen, ACCENT, canvas_rect.inflate(10, 10), border_radius=12, width=2
    )
    screen.blit(canvas, canvas_rect)
    screen.blit(left_txt, (40, HEIGHT - 50))

    # Right panel (Predictions and preview)
    pred_x = CANVAS_PANEL_WIDTH + 40
    for i, p in enumerate(probs):
        y_pred = 80 + i * 32
        color = ACCENT if i == pred else BLUE
        draw_prob_bar(
            screen,
            pred_x + 30,
            y_pred,
            PROB_BAR_WIDTH,
            PROB_BAR_HEIGHT,
            p,
            color,
            str(i),
        )

    # Prediction display
    pygame.draw.rect(screen, PANEL_COLOR, result_bg, border_radius=20)
    if pred is not None:
        res_txt = font_bold.render(str(int(pred)), True, ACCENT_STRONG)
        res_rect = res_txt.get_rect(center=result_bg.center)
        screen.blit(res_txt, res_rect)

    # 28x28 Preview (What the model sees)
    pygame.draw.rect(screen, MUTED, (preview_pos[0] - 1, preview_pos[1] - 1, 30, 30), 1)
    screen.blit(scaled_canvas, preview_pos)

    pygame.display.update()
    clock.tick(FPS)
