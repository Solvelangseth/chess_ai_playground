# main.py
import pygame
import sys
from game import Game
from settings import SCREEN_WIDTH, SCREEN_HEIGHT

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Chess with AI")
    game = Game(screen)
    game.run()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
