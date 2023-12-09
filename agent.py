"""
Agent module
"""
from random import randint, random
import numpy as np

def q_learning(environment, learning_rate, gamma, total_iteration, show=False):
    """
    Q-learning: An off-policy TD control algorithm
    as described in Reinforcement Learning: An Introduction" 1998 p158 by Richard S. SUTTON
    https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

    Some comments include ##'comment', they quote from Richard S. SUTTON pseudo-code's
    """

    #ADDED features compared to Richard S. SUTTON algoritm's
    epoch_show = 200
    exploration_function = lambda i: -1 / (total_iteration * 0.8) * i + 1

    ##'Initialize Q(s,a)...arbitrarly execpt that Q(terminal,.)=0'
    number_states = environment.nb_discretisation_x
    number_action = 2

    Q = np.random.rand(number_states, number_states, number_states, number_states, number_action)


    ##'Repeat (for each episode):'
    for iteration in range(total_iteration):
        end = False
        ##'Initialize S'
        statut = environment.reset()

        ##'Repeat (for each step of episode):'
        while not end:

            ##'Choose A from S using ...'
            if random() < exploration_function(iteration):
                 #ADDED: features to encourage erratic behavior
                action = randint(0, number_action-1)
            else:
                 ##'policy derived from Q'
                action = np.argmax(Q[statut[0], statut[1], statut[2], statut[3]])

            ##'Take action  A ...'
            observation = environment.step(action)

            ##', observe R,S''
            futur_statut, reward, end = observation

            ##'Q(S,A)=Q(S,A)+learning_rate*...'
            Q[statut[0], statut[1], statut[2], statut[3], action] = Q[statut[0], statut[1], statut[2], statut[3], action] + learning_rate * (reward + gamma * np.max(Q[futur_statut[0], futur_statut[1], futur_statut[2], futur_statut[3], :]) - Q[statut[0], statut[1], statut[2], statut[3], action])

            ##'S becomes S''
            statut = futur_statut.copy()

            #ADDED: Show behavior in a window
            if (iteration%epoch_show == 0 or iteration == total_iteration-1) and show:
                environment.render()
    return Q
