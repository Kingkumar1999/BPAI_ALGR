#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np
import random
import math
import threading
from itertools import product
import sys
import os

import robobo
# import MLPbrain
import cv2
import sys
import signal
import prey

geneBank = {}

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
        
    for L in [100, 150, 200]:
        for I in [15, 20, 25]:
            for _ in range(10):

                time.sleep(10)
                # ////
                # Start simulation
                signal.signal(signal.SIGINT, terminate_program)
                rob = robobo.SimulationRobobo(L, I).connect(address='192.168.56.1', port=19997)
                rob.play_simulation()
                
                # Log initial data
                variant = rob.get_name()

                individual_path = uniquify("Individuals_{}.txt".format(variant))
                file_indiv = open(individual_path, "w")
                file_indiv.write("[time, indiv, parent1, parent2, gender, brain]"+ "\n")

                position_path = uniquify("Positions_{}.txt".format(variant))
                file_locs = open(position_path, "w")
                file_locs.write("[time, indiv, gender, coords]" + '\n')

                np.set_printoptions(threshold=sys.maxsize)

                # /////
                # Start the initial population
                current_pop = []
                currenttime = rob.get_sim_time()
                parent1 = parent2 = -1
                for x in range(rob.init_pop_size):
                    indiv = rob.birth()
                    brain = rob.createBrain()
                    gender = x%2
                    geneBank[indiv] = brain
                    current_pop.append([indiv, gender, brain, 0])
                    rob.change_colour(indiv, gender)
                    rob.population[indiv].gender = gender
                    
                    file_indiv.write(str([currenttime, indiv, parent1, parent2, gender, brain.toString()])+ "\n")
                file_indiv.flush()

                time.sleep(1)

                # /////
                # Track metrics and check for convergence
                pairings = set()
                
                age = 1
                queue = []

                while(True):
                    print("age {}".format(age))

                    if len(current_pop) < 2:
                        result = "Extinction!"
                        break
                    else:

                        # moving the bots
                        for individual in current_pop:
                            inputs = rob.useCamera(individual[0], individual[1])
                            inputs = rob.prepInputs(inputs)
                            outputs = individual[2].forward(inputs)
                            rob.move(individual[0], outputs[0]*rob.speed, outputs[1]*rob.speed, rob.move_time)
                            individual[3] += 1

                        # running checks
                        if age % 2 == 0:
                            print("ding")
                            file_locs.write(str([rob.get_sim_time(), "popsize", len(current_pop), "queuesize", len(queue)]) + '\n')
                            # Fetching data
                            current_data = [[], []]
                            for individual in current_pop.copy():
                                
                                if (individual[3] != rob.lifespan):
                                    current_data[individual[1]].append((individual[0], list(rob.position(individual[0])[1][0:2])))
                                else:
                                    rob.move(individual[0], 0, 0, 2)
                                    rob.death(individual[0])
                                    geneBank.pop(individual[0])
                                    rob.change_colour(individual[0], 3)
                                    file_indiv.write(str([rob.get_sim_time(), individual[0], "death"]) + '\n')
                                    current_pop.remove(individual)

                            # Log current data
                            currenttime = rob.get_sim_time()
                            for gender in range(len(current_data)):
                                for pair in current_data[gender]:
                                    file_locs.write(str([currenttime, pair[0], gender, pair[1]]) + '\n')
                            file_locs.flush()

                            # Reproduction
                            for male, female in product(current_data[0], current_data[1]):
                                if (math.dist(male[1], female[1]) <= rob.range):
                                    print("{} met {}".format(male[0], female[0]))
                                    #  something is born
                                    choices = [male[0], female[0]]
                                    choice_set = frozenset((id(geneBank[choices[0]]), id(geneBank[choices[1]])))
                                    if (choice_set not in pairings):
                                        pairings.add(choice_set)
                                        queue.append( spawnbot(rob, choices))
                                    else:
                                        print("we broke up")


                            currenttime = rob.get_sim_time()
                            while(len(current_pop) != rob.max_pop_size and len(queue) > 0):
                                nextChild = queue.pop(0) # [choices, gender, brain, age]

                                indiv = rob.birth()
                                geneBank[indiv] = nextChild[2]
                                
                                rob.change_colour(indiv, nextChild[1])
                                rob.population[indiv].gender = nextChild[1]
                                print("Creating child with parents {}".format(nextChild[0]))
                                
                                current_pop.append( [indiv, nextChild[1], nextChild[2], nextChild[3]])
                                file_indiv.write(str([currenttime, indiv, "birth", nextChild[0][0], nextChild[0][1], nextChild[1], nextChild[2].toString()]) + '\n')
                        age +=1

                    # pause to time can pass without flooding the processor
                    # time.sleep(1)

                currenttime = rob.get_sim_time()

                for individual in current_pop:
                    rob.move(individual[0], 0, 0, 2)
                    rob.death(individual[0])
                    rob.change_colour(individual[0], 3)
                    file_indiv.write(str([currenttime, individual[0], "death"]) + '\n')

                print(result)
                file_indiv.write(str([currenttime, result]) + '\n')
                        

                # /////
                # End sim

                rob.stop_world()



def spawnbot(sim, choices): 
    
    new_brain = robobo.MLPbrain.crossover(geneBank[choices[0]], geneBank[choices[1]])
    new_brain.mutate(sim.mutation_rate)
    gender = random.randint(0,1)
    
    return [choices, gender, new_brain, 0]

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


if __name__ == "__main__":
    main()
