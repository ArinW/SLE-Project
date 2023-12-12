"""
The Inverted Pendulum Balancing Robot by Julien STEYER
July 2019
"""

from environment import CartPoleEnv
from agent import q_learning
import matplotlib.pyplot as plt
environment1 = CartPoleEnv()
list1 = []
list2 = []
list3 = []
cumulative_reward = 0
for x in range(2000, 5000, 1000):
    Q, total_reward, successful_episodes = q_learning(environment1, learning_rate=0.05, gamma=0.99, total_iteration=x, show=True)
    cumulative_reward += total_reward
    list1.append(x)
    list2.append(total_reward/x)
    list3.append(cumulative_reward)
environment1.close()
print(list1)
print(list2)
print(list3)
plt.figure(1)
plt.axis([min(list1), max(list1), 0, max(list2)])
plt.plot(list1, list2)
plt.title("Outcome 1")
plt.xlabel("total_iteration")
plt.ylabel("total reward received per episode")
plt.show()
plt.figure(2)
plt.axis([min(list1), max(list1), 0, max(list3)])
plt.plot(list1, list3)
plt.title("Outcome 2")
plt.xlabel("total_iteration")
plt.ylabel("cumulative_reward")
plt.show()
for episode_info in successful_episodes:
    episode_number = episode_info['episode']
    x_values = episode_info['x_values']
    theta_values = episode_info['theta_values']
    num_steps = episode_info['num_steps']
    time_steps = range(num_steps)

   # Plot x
plt.figure(3)
plt.plot(time_steps, x_values)
plt.title(f"Episode {episode_number} - Cart Position (x) over Time")
plt.xlabel("Time Steps")
plt.ylabel("Cart Position (x)")
plt.show()

    # Plot θ
plt.figure(4)
plt.plot(time_steps, theta_values)
plt.title(f"Episode {episode_number} - Pole Angle (θ) over Time")
plt.xlabel("Time Steps")
plt.ylabel("Pole Angle (θ)")
plt.show()
