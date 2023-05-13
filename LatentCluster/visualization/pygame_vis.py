from typing import Any, Iterable

import numpy as np
import pygame

# NOTE: Code below is 90% generated

# Game that renders a bunch of points associated with a string (strings contained in text). Clicking a point shows
# the text associated with it. Added functionality to zoom in and out and move the zoomed window with WASD.
# Mode can be text or image
class PointVis:
    def __init__(
        self,
        points: Iterable[np.ndarray], data: Iterable[Any],
        width: int, height: int, colors : Iterable[np.ndarray] = None,
        mode = "TEXT"
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))

        self.mode = mode

        self.screen_width = width
        self.screen_height = height

        self.points = points
        self.colors = colors

        # Rescale points so they fit on screen
        x = self.points[:,0]
        y = self.points[:,1]
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

        self.points[:,0] = x
        self.points[:,1] = y

        self.texts = data
        self.images = data

        self.font = pygame.font.Font(None, 36)
        self.databox = None

        if self.mode == "TEXT":
            self.databox_surface = self.font.render(self.texts[0], True, (255, 255, 255))
        else:
            self.databox_surface = None

        self.clicked_index = None

        self.POINT_RADIUS = 3
        self.ZOOM_FACTOR = 0.998
        self.MOVE_FACTOR = 0.0001

        # Offsets for the view window
        self.offset_x = -1 * (abs(self.points[:,0].max()) + abs(self.points[:,0].min()))/2 
        self.offset_y = -1 * (abs(self.points[:,1].max()) + abs(self.points[:,1].min()))/2 

        # Current width and height of the view window (will change)
        self.w = 2
        self.h = 2

        self.run()
    
    def in_view(self, x, y):
        return True
        return (x >= self.offset_x and x <= self.offset_x + self.w) and \
            (y >= self.offset_y and y <= self.offset_y + self.h)

    def draw_points(self):
        width, height = self.screen.get_size()
        for i, transformed_point in enumerate(self.transform_points(self.points)):
            if self.in_view(transformed_point[0], transformed_point[1]):
                pygame.draw.circle(self.screen, (255, 255, 255) if self.colors is None else self.colors[i], transformed_point, self.POINT_RADIUS)

    def draw_textbox(self, index):
        transformed_point = self.transform_points(self.points[index][None,:])[0]

        self.text_surface = self.font.render(self.texts[index], True, (255, 255, 255))
        self.databox_surface = self.text_surface  # Update the textbox_surface
        text_rect = self.text_surface.get_rect()
        text_rect.center = transformed_point
        pygame.draw.rect(self.screen, (0, 0, 0), text_rect.inflate(10, 10))
        self.screen.blit(self.text_surface, text_rect)
        self.databox = text_rect

    def draw_image(self, index):

        image = self.images[index]
        mode = image.mode
        size = image.size
        data = image.tobytes()

        image_surface = pygame.image.fromstring(data, size, mode)
        image_rect = image_surface.get_rect()
        image_rect.topleft = (0, 0)
        pygame.draw.rect(self.screen, (0, 0, 0), image_rect.inflate(10, 10))
        self.screen.blit(image_surface, image_rect)
        self.databox = image_rect

    def check_click(self, pos):
        for index, transformed_point in enumerate(self.transform_points(self.points)):
            if (transformed_point[0] - pos[0]) ** 2 + (transformed_point[1] - pos[1]) ** 2 <= 5 ** 2:
                return index
        return None

    def transform_points(self, points):
        points_new = points - np.array([self.offset_x, self.offset_y])[None,:]

        # Gives us coordinates with respect to the view window. We now want to normalize such that
        # [0, self.w] -> [0, self.screen_width] and [0, self.h] -> [0, self.screen_height]
        points_new[:,0] *= (self.screen_width / self.w)
        points_new[:,1] *= (self.screen_height / self.h)
        return points_new

    def run(self):
        running = True
        while running:
            self.screen.fill((0, 0, 0))
            self.draw_points()
            if self.databox and self.databox_surface:
                self.screen.blit(self.databox_surface, self.databox)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.clicked_index = self.check_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        width, height = self.screen.get_size()

                        self.w = width
                        self.height = h
                        self.offset_x = 0
                        self.offset_y = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] or keys[pygame.K_e]:
                scale = 1
                if keys[pygame.K_q]:
                    scale *= self.ZOOM_FACTOR
                if keys[pygame.K_e]:
                    scale /= self.ZOOM_FACTOR

                self.w *= scale
                self.h *= scale
                
                mult = (1 - 1/scale) # w_new - w_old = w_new - (w_new/scale) = (w_new) * (1 - 1/scale)
                delta_w = self.w * mult
                delta_h = self.h * mult

                self.offset_x += delta_w / 2
                self.offset_y += delta_h / 2

            if keys[pygame.K_w]:
                self.offset_y -= 10 * self.MOVE_FACTOR * self.w
            if keys[pygame.K_a]:
                self.offset_x -= 10 * self.MOVE_FACTOR * self.w
            if keys[pygame.K_s]:
                self.offset_y += 10 * self.MOVE_FACTOR * self.w
            if keys[pygame.K_d]:
                self.offset_x += 10 * self.MOVE_FACTOR * self.w

            if self.clicked_index is not None:
                if self.mode == "TEXT":
                    self.draw_textbox(self.clicked_index)
                elif self.mode == "IMAGE":
                    self.draw_image(self.clicked_index)
            else:
                self.databox = None

            pygame.display.flip()

        pygame.quit()

