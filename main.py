#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:54:11 2025

@author: robertopitz
"""

import numpy as np
from utils import get_prey_pos_list
from botClass import botClass
from trainBot import train
from playTestGame import play_test_game

# set board
board = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]

# trasform to numpy array
board = np.array(board)

# set activation functions
def relu(x):
    return np.maximum(0.0, x)

def linear(x):
    return x

# set ANN layout
ann_layout = [{"n": 100, "act_func": relu},
              #{"n": 5, "act_func": relu},
              #{"n": 5, "act_func": relu},
              {"n": 2, "act_func": linear}]

# get bot instance
bot = botClass(bot_pos    = np.array([2, 2]),
               ann_layout = ann_layout)

# prepare list with prey positions for training
print("--create prey_pos_list--")
prey_pos_list = get_prey_pos_list(int(1e4), bot.bot_pos, board)

# set maximum iterations    
max_iter = int(100e6)

# train bot
print("--train--")
trained_bot = train(bot, board, prey_pos_list, max_iter)

# play a test game using pygame
print("--play test game--")
play_test_game(trained_bot, board)
