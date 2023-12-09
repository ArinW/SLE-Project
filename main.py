"""
The Inverted Pendulum Balancing Robot by Julien STEYER
July 2019
"""

from environment import CartPoleEnv
from agent import q_learning

environment1 = CartPoleEnv()

q_learning(environment1, learning_rate=0.06, gamma=0.99, total_iteration=20000, show=True)

environment1.close()
