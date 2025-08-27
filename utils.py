#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:50:25 2025

@author: robertopitz
"""

import numpy as np
from random import randrange

def get_prey_pos_list(n: int, pos: list, board: list):
    prey_pos_list = np.zeros( 2*n ).reshape((n, 2))
    prey_pos_list[0] = get_new_prey_pos(pos, board)
    for i in range(1, n):
        prey_pos_list[i] = get_new_prey_pos(prey_pos_list[i-1], board)
    return prey_pos_list


def get_new_prey_pos(pos, board):
    while True:
        c = randrange(1, len(board)-1)
        r = randrange(1, len(board[0])-1)
        if c != pos[0] or r != pos[1]:
            if board[c][r] == 0:
                return np.array([c, r])