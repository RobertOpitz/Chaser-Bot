#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:16:13 2025

@author: robertopitz
"""

import numpy as np
import time

def train(bot_mother, board, prey_pos_list, max_iter):

    max_num_steps = int(np.sum(0.5*(1-board)))
    print("maximum steps the bot should do:", max_num_steps)
    
    j_old = 0
    new_innovation_count = 0
    length_prey_list = np.size(prey_pos_list, 0)

    max_catched_prey = play_train_game(bot_mother, prey_pos_list,
                                       board, max_num_steps)

    print("initially catched prey before training:", max_catched_prey)
    
    start = time.time()#perf_counter()
    
    start_round = time.time()

    print("=== START ===")
    for j in range(1, max_iter):
        
        #start_round = time.time()

        # create mutated daughter from mother
        bot_daughter = bot_mother.deep_copy()#.mutate()
        bot_daughter.mutate()

        # play the game
        catched_prey = play_train_game(bot_daughter, 
                                       prey_pos_list,
                                       board, 
                                       max_num_steps)

        # check results of played game to see, if daughter was as good,
        # or better than mother
        if catched_prey > max_catched_prey:
            # daughter is better than mother
            # dautghter becomes the new mother
            bot_mother = bot_daughter.deep_copy()

            max_catched_prey = catched_prey

            print("=== Improvement after", j, ". Iterations ===")
            new_innovation_count += 1
            print(new_innovation_count, ". innovation")
            print("max catched prey: ", max_catched_prey)
            print('Nb. of iterations since last increase: ', j - j_old)
            #print("Found after:",
            #      time.strftime("%H:%M:%S", 
            #                    time.gmtime((time.perf_counter() - start))))
            print("Found after:", round(time.time() - start_round, 2), "sec")
            print('')

            start_round = time.time()#perf_counter()
            j_old = j

            if max_catched_prey == length_prey_list:
                break

        elif catched_prey == max_catched_prey:
            # daughter is as good as mother
            # daughter becomes the new mother
            bot_mother = bot_daughter.deep_copy()
            
    print("Final:", round(time.time() - start, 1), "sec")

    return bot_mother

def play_train_game(bot, prey_pos_list, board, max_num_steps):

    #bot_pos = np.copy(bot_pos_start)
    catched_prey = 0

    for prey_pos in prey_pos_list:

        prey_is_catched = False
        for _ in range(1, max_num_steps):
            bot.move_bot(prey_pos, board)

            # check bot
            if bot.bot_pos[0] == prey_pos[0] and bot.bot_pos[1] == prey_pos[1]:
                catched_prey += 1
                prey_is_catched = True
                break

        if not prey_is_catched:
            break

    return catched_prey

