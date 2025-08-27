#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 09:21:39 2025

@author: robertopitz

bot class
"""
import numpy as np
import math
from copy import deepcopy

rng = np.random.default_rng(42)

class botClass:
    def __init__(self, bot_pos: list, ann_layout: dict, nb_input: int = 12):
        
        # set initial position of bot
        self.bot_pos = bot_pos.copy()
        
        # set ann
        self.ann = self._set_ann(nb_input = nb_input, ann_layout = ann_layout)
        
        
    def deep_copy(self):
        return deepcopy(self)
        
    def move_bot(self, prey_pos, board):

        # get input vector x for bot
        x = self._get_x(prey_pos, board)
        
        move = self._next_move_ai(x)
        
        c = self.bot_pos[0]
        r = self.bot_pos[1]
        
        if move == "UP":
            if board[c][r-1] == 0:
                self.bot_pos[1] -= 1
        elif move == "DOWN":
            if board[c][r+1] == 0:
                self.bot_pos[1] += 1
        elif move == "LEFT":
            if board[c-1][r] == 0:
                self.bot_pos[0] -= 1
        elif move == "RIGHT":
            if board[c+1][r] == 0:
                self.bot_pos[0] += 1
                
    def mutate(self):

        mutated = False
        while not mutated:
            mutated_ann = []
            for layer in self.ann:
                # create temporary fallten version of this layer
                tmp_org_shape = layer["weights"].shape
                tmp = np.copy(layer["weights"]).flatten()
                # mutate tmp
                # 1.) how many mutation in this layer?
                n = rng.integers(low  = 0,
                                 high = 4,
                                 size = 1)
                if n > 0:
                    mutated = True
                    # 2.) where to introduce the mutations?
                    # list with positions of weights to be mutated
                    m = rng.integers(low  = 0,
                                     high = tmp.size,
                                     size = n)
                    # 3.) set new random weight to the selected weights
                    tmp[m] = np.random.laplace(loc   = 0,
                                               scale = 4,
                                               size  = n)#.round(decimals = 1)
                # build new mutated layer
                mutated_ann.append({"weights": tmp.reshape(tmp_org_shape),
                                    "act_func": layer["act_func"]})
                
        self.ann = mutated_ann
                    
    def _set_ann(self, nb_input: int, ann_layout: dict):
                
        ann = []
        c = nb_input + 1
        for layer in ann_layout:

            r = layer["n"]
            n = c * r
            
            init_weights = np.random.uniform(low  = -1,
                                             high =  1,
                                             size = n)
            
            new_layer = {"weights": np.array(init_weights).reshape((r, c)),
                         "act_func": layer["act_func"]}
            
            c = r + 1
            
            ann.append(new_layer)
        
        return ann
    
    def _next_move_ai(self, x):

        # compute x
        for layer in self.ann:
            # add bias
            x = np.append(x, 1.0)
            # do linear part Wx
            x = np.matmul(layer["weights"], x)
            # do non-linear part
            x = layer["act_func"](x)

        # evaluate ann result
        v = x[0]
        u = x[1]
        if math.fabs(v) > math.fabs(u):
            if v < 0:
                this_step = "UP"
            else:
                this_step = "DOWN"
        else:
            if u < 0:
                this_step = "LEFT"
            else:
                this_step = "RIGHT"

        return this_step
    
    def _get_x(self, prey_pos, board):
        
        # get board wall features
        c = self.bot_pos[0]  # column position of bot
        r = self.bot_pos[1]  # row position of bot
        board_walls = np.array([board[c+1][r-1], board[c+1][r], board[c+1][r+1],
                                board[c  ][r-1],                board[c  ][r+1],
                                board[c-1][r-1], board[c-1][r], board[c-1][r+1]])
        
        board_walls = 2.0 * board_walls - 1.0

        # set features that the bot gets
        x = self.bot_pos
        x = np.append(x, prey_pos)
        x = np.append(x, board_walls)

        return x

