#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:45:15 2025

@author: robertopitz

play test game with pygame

"""

import pygame as pg
import numpy as np

from utils import get_new_prey_pos

def draw_board(screen, board, rs):
    for c in range(len(board)):
        for r in range(len(board[0])):
            if board[c][r] == 1:
                pg.draw.rect(screen,
                             pg.Color("blue"),
                             pg.Rect(c * rs,
                                     r * rs,
                                     rs, rs))


def draw_bot(screen, pos, rs):
    pg.draw.rect(screen,
                 pg.Color("red"),
                 pg.Rect(pos[0] * rs,
                         pos[1] * rs,
                         rs, rs))


def draw_prey(screen, pos, rs):
    pg.draw.rect(screen,
                 pg.Color("yellow"),
                 pg.Rect(pos[0] * rs,
                         pos[1] * rs,
                         rs, rs))


def play_test_game(bot, board):

    rect_size = 15
    #bot = np.copy(bot_pos_start)

    pg.init()
    screen_color = pg.Color("black")

    screen = pg.display.set_mode((np.size(board, 0) * rect_size,
                                  np.size(board, 1) * rect_size))
    clock = pg.time.Clock()

    pg.display.set_caption("Clean Bot AI")

    running = True
    prey_pos = get_new_prey_pos(bot.bot_pos, board)
    while running:

        bot.move_bot(prey_pos, board)

        if bot.bot_pos[0] == prey_pos[0] and bot.bot_pos[1] == prey_pos[1]:
            prey_pos = get_new_prey_pos(bot.bot_pos, board)

        screen.fill(screen_color)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        draw_board(screen, board, rect_size)
        draw_prey(screen, prey_pos, rect_size)
        draw_bot(screen, bot.bot_pos, rect_size)

        clock.tick(60)

        pg.display.flip()

    pg.quit()