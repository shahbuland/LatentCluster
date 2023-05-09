from typing import Any, Iterable

import numpy as np
import pygame

# NOTE: Code below is 90% generated

# Game that renders a bunch of points associated with a string (strings contained in text). Clicking a point shows
# the text associated with it. Added functionality to zoom in and out and move the zoomed window with WASD.
class PointTextVis:
    def __init__(self, points: Iterable[np.ndarray], texts: Iterable[str], width: int, height: int, colors : Iterable[np.ndarray] = None):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.points = points
        self.colors = colors

        # Rescale points so they fit on screen
        self.points[:, 0] *= (width / self.points[:, 0].max())
        self.points[:, 1] *= (height / self.points[:, 1].max())

        self.texts = texts
        self.font = pygame.font.Font(None, 36)
        self.textbox = None
        self.textbox_surface = self.font.render(self.texts[0], True, (255, 255, 255))

        self.clicked_index = None

        self.POINT_RADIUS = 3
        self.ZOOM_FACTOR = 0.025
        self.MOVE_FACTOR = 1.0

        self.zoom = 1
        self.offset_x = 0
        self.offset_y = 0

        self.run()

    def draw_points(self):
        width, height = self.screen.get_size()
        for i, point in enumerate(self.points):
            transformed_point = self.transform_point(point)
            if 0 <= transformed_point[0] <= width and 0 <= transformed_point[1] <= height:
                pygame.draw.circle(self.screen, (255, 255, 255) if self.colors is None else self.colors[i], transformed_point, self.POINT_RADIUS)

    def draw_textbox(self, index):
        transformed_point = self.transform_point(self.points[index])
        self.text_surface = self.font.render(self.texts[index], True, (255, 255, 255))
        text_rect = self.text_surface.get_rect()
        text_rect.center = transformed_point
        pygame.draw.rect(self.screen, (0, 0, 0), text_rect.inflate(10, 10))
        self.screen.blit(self.text_surface, text_rect)
        self.textbox = text_rect

    def check_click(self, pos):
        for index, point in enumerate(self.points):
            transformed_point = self.transform_point(point)
            if (transformed_point[0] - pos[0]) ** 2 + (transformed_point[1] - pos[1]) ** 2 <= 5 ** 2:
                return index
        return None

    def transform_point(self, point):
        return int(point[0] * self.zoom + self.offset_x), int(point[1] * self.zoom + self.offset_y)

    def run(self):
        running = True
        while running:
            self.screen.fill((0, 0, 0))
            self.draw_points()
            if self.textbox:
                self.screen.blit(self.textbox_surface, self.textbox)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.clicked_index = self.check_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.zoom = 1
                        self.offset_x = 0
                        self.offset_y = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] or keys[pygame.K_e]:
                old_zoom = self.zoom
                old_center_x = self.screen.get_width() / 2 + self.offset_x / old_zoom
                old_center_y = self.screen.get_height() / 2 + self.offset_y / old_zoom

                if keys[pygame.K_q]:
                    self.zoom *= (1 - self.ZOOM_FACTOR)
                if keys[pygame.K_e]:
                    self.zoom *= (1 + self.ZOOM_FACTOR)

                new_center_x = self.screen.get_width() / 2 + self.offset_x / self.zoom
                new_center_y = self.screen.get_height() / 2 + self.offset_y / self.zoom

                self.offset_x += (old_center_x - new_center_x) * self.zoom
                self.offset_y += (old_center_y - new_center_y) * self.zoom

            if keys[pygame.K_w]:
                self.offset_y += 10 * self.MOVE_FACTOR
            if keys[pygame.K_a]:
                self.offset_x += 10 * self.MOVE_FACTOR
            if keys[pygame.K_s]:
                self.offset_y -= 10 * self.MOVE_FACTOR
            if keys[pygame.K_d]:
                self.offset_x -= 10 * self.MOVE_FACTOR

            if self.clicked_index is not None:
                self.draw_textbox(self.clicked_index)
            else:
                self.textbox = None

            pygame.display.flip()

        pygame.quit()