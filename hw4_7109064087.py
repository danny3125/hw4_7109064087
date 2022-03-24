#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:13:09 2022

@author: wangyiren
"""
from cat_and_mouse import Cat_and_Mouse
import numpy as np


def monte_carlo_format(random,monte_carlo,cm,showup):
    policy = np.zeros((cm.numStates)) #A pi(s)
    q_table = np.zeros((cm.numStates,cm.numActions))
    policy = np.argmax(q_table, axis=1)
    returns_times = np.zeros((cm.numStates,cm.numActions))
    episolon = 1
    decrease_number = 0.99
    gamma = 0.9
    for j in range(random + monte_carlo):
        returns = np.zeros((cm.numStates,cm.numActions))
        gameOver=False
        cm.reset()
        trajectory = []
        G = 0
        t = 0
        i=0
        while not gameOver:
            episolon_greedy = [episolon/cm.numActions]*cm.numActions
            episolon_greedy[policy[cm.currentState()]] = 1 - episolon + (episolon/cm.numActions)
            action = np.random.choice(cm.numActions,1,p=episolon_greedy)
            current_state = cm.currentState()
            (nextState,reward,gameOver)=cm.step(action[0]) #take the action, move to next state, then move to next step
            trajectory.append((current_state,action[0],reward,nextState))
            if j == (monte_carlo-1):
                cm.render()
                #cm.render(outfile= showup +str(i)+'.png')   #render current state of the environment to .png file
            i+=1   
            t+=1
        for step in range(1,t+1):
            G =  gamma*G + trajectory[t-step][2]
            if returns[trajectory[t-step][0]][trajectory[t-step][1]]==0:
                q_table[trajectory[t-step][0]][trajectory[t-step][1]]\
                *= returns_times[trajectory[t-step][0]][trajectory[t-step][1]]
                returns_times[trajectory[t-step][0]][trajectory[t-step][1]] +=1
                returns[trajectory[t-step][0]][trajectory[t-step][1]] = G
                q_table[trajectory[t-step][0]][trajectory[t-step][1]] += G
                q_table[trajectory[t-step][0]][trajectory[t-step][1]] \
                /= returns_times[trajectory[t-step][0]][trajectory[t-step][1]]
                policy = np.argmax(q_table, axis=1)
        if episolon > 0.001:
            episolon *= decrease_number
    return policy            
 

cm=Cat_and_Mouse(rows=1, columns=7, mouseInitLoc=[0,3],
                 cheeseLocs=[[0,0],[0,6]], stickyLocs=[[0,2]],
                 slipperyLocs=[[0,4]])

print("part one 1D:")
print('NumStates: {}'.format(cm.numStates))
print('NumActions: {}'.format(cm.numActions))
policy = monte_carlo_format(300, 0, cm,'part one 1D')
print('policy: ',policy)
print("part two 1D:")
print('NumStates: {}'.format(cm.numStates))
print('NumActions: {}'.format(cm.numActions))
policy = monte_carlo_format(300, 300, cm, 'part two 1D')
print('policy: ',policy)
cm=Cat_and_Mouse(rows=5,columns=5,mouseInitLoc=[0,0],stickyLocs=[[2,4],[3,4]],slipperyLocs=[[1,1],[2,1]],\
            cheeseLocs=[[4,4]],catLocs=[[3,2],[3,3]])   
print("part one 2D:")
print('NumStates: {}'.format(cm.numStates))
print('NumActions: {}'.format(cm.numActions))
policy = monte_carlo_format(300, 0, cm, 'part one 2D')
print('policy: \n',policy.reshape((5,5)))
print("part two 2D:")
print('NumStates: {}'.format(cm.numStates))
print('NumActions: {}'.format(cm.numActions))
policy = monte_carlo_format(300000, 600000, cm, 'part two 2D')
print('policy: \n',policy.reshape((5,5)))
    