
import pygame

DRAW_MULT = 0.05
MAX_X = 16000
MAX_Y = 9000


WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,  0)
BLACK = (  0,   0,  0)
(width, height) = (int(MAX_X * DRAW_MULT) , int(MAX_Y * DRAW_MULT))


CP_RAD = 600
POD_RAD = 400


class Draw:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TUFF")

    def Draw(self, game):
        self.screen.fill(BLACK)
        for cp in game.map:
            self.DrawC(cp[0], cp[1], CP_RAD, RED)


        tarCp = game.map[game.pod.cp % len(game.map)]
        self.DrawC(tarCp[0],tarCp[1], CP_RAD, BLUE)

        self.DrawC(game.pod.pos[0], game.pod.pos[1], POD_RAD, GREEN)
        self.DrawC(game.blocker.pos[0], game.blocker.pos[1], POD_RAD, (255, 165, 0))

        pygame.display.update()
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                running = False

    def DrawC(self, x, y, r, c):
        pygame.draw.circle(self.screen, c, (int( (x + 8000) / 2 * DRAW_MULT), int( (y + 4500 ) / 2 * DRAW_MULT)), int(r * DRAW_MULT / 2))



