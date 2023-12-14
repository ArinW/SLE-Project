## Contents:
0. [Installation](#installation)
1. [Intoduction](#introduction)
2. [The agent](#agent)
3. [The environment](#environment)
4. [Files structure](#file)



<a id='installation'></a>
## Installation
1. Run the program
```
$ py main.py
```



<a id='introduction'></a>
## Introduction

<p align="justify">The objective of this project is to teach a robotic cart to balance an inverted pendulum by moving the cart along a linear axis. The agent in this case the cart, will have to be able to understand through the acquired experience the combinations of actions to be performed depending on the state of the pendulum to keep the pole in its unstable equilibrium position.

The objective being to achieve [this type of behavior.](https://www.youtube.com/watch?v=Lt-KLtkDlh8)

This will be achieved using a reinforcement learning algorithm.</p>

<a id='agent'></a>
### The agent
<p align="justify">The goal of the agent, or more specifically of the algorithm that governs his behavior, is to maximize the value of the rewards obtained. To do so, it records a weighted average of the rewards obtained according to the actions At taken in the environment at the state St.

Among the many reinforcement learning algorithms available in the literature, I chose the <b>Q-Learning</b> algorithm. It has the advantage of being very popular, easy to use and therefore very well documented. It is used in its simplest form, without using deep learning technics to be easier to understand although more limited.

The Q-Learning algorithm works using a matrix called Q-table, which has each doublet of state and action (St,At) associated with the weighted average of the rewards obtained.</p>

<a id='environment'></a>
### The environment
<p align="justify">It was also necessary to model the environment in which the agent evolves, in our case a two-dimensional physical model of the pendulum's behavior.
It can be obtained simply by applying <i>"Newton's Laws of Motion"</i>.</p>

<p align="center"><img style="display: block; margin: auto;" src="image/model.jpg" /><br>
<i>Figure 4 - Physical model used</i></p>

<p align="justify">
These equations are solved numerically using the <i>"Euler Method"</i>.

In order to be able to visualize the behavior of the cart, a modified version of the "CartPole-v1" environment from the gym library is used.</p>


<a id='file'></a>
## File structure

### main .py
The program can be started by simply running 'main.py', but the library <i>numpy</i> must be installed first.
It is possible to modify the hyperparameters of the algorithm by modifying the variables contained in the 'main.py' file.

```python
#Settings for Qlearning's algorithm
for x in range(2000, 4000, 1000):
    Q, total_reward, successful_episodes = q_learning(environment1, learning_rate=0.05, gamma=0.99, total_iteration=x, show=True)
```


### environment .py
The 'environment.py' file contains several methods that are directly extracted from the *CartPole-V1* environment of the *gym* library created by OpenAI.
Some passages have been modified so that you have direct access to the equations of the motion of the inverted pendulum.

```python
#Settings for Qlearning's algorithm
self.X = np.linspace(-self.x_limit, self.x_limit, self.nb_discretisation_x)
self.X_dot = np.linspace(-self.x_dot_limit, self.x_dot_limit, self.nb_discretisation_x_dot)
self.THETA = np.linspace(-self.theta_limit, self.theta_limit, self.nb_discretisation_theta)
self.THETA_dot = np.linspace(-self.theta_dot_limit, self.theta_dot_limit, self.nb_discretisation_theta_dot)
```

Several variables are also very important for environmental modeling, including the declaration of reward zone boundaries, the value of associated rewards, and the number of discretization states.

```python
#Settings for borders
self.gravity = 9.8
self.masscart = 1.0
self.masspole = 0.1
self.length = 0.5        #(half the size)
self.force_mag = 10.0
self.mu_c = 0.0005
self.mu_p = 0.000002
#Setting for numerical calcul
self.tau = 0.02         #setting for Euler's method
self.dt_system = 0.02    #reaction time of the systeme (for real application)
#Settings for borders
self.x_limit = 2.4                         #maximum position
self.theta_limit = 12 * 2 * pi / 360   #maximum angle
self.x_dot_limit = 0.5                     #maximum linear velocity
self.theta_dot_limit = 50                 #maximum angular velocity
#Others settings
self.nb_discretisation = 11           #number of discretisation for each of the 4 states
self.Recompenses = [-1, 0, 0, 1]   #reward for [near to border, neutral, vertical pole, vertical pole and cart in the center]
```


### agent.py
The *agent.py* file contains the Q-Learning algorithm which is based on the pseudocode available in *Sutton, R. S., Barto, A. G., Reinforcement Learning: An Introduction [archive]. MIT Press, 1998. 2e édition MIT Press en 2018.*

Some lines present additional instructions, they have only a technical use, either to display the environment or simply to express logic in python language.
The only passage with a different logic is the one allowing the application of the exploration/exploitation principle.

```python
##'Choose A from S using ...'
if random() < exploration_function(iteration):
     #ADDED: features to encourage erratic behavior
     action = randint(0, number_action-1)
else:
     ##'policy derived from Q'
     action = np.argmax(Q[statut[0], statut[1], statut[2], statut[3]])
```
We record the most successful one in the cycle.
```python
successful_episodes.append({
    'episode': iteration,
    'x_values': episode_states,
    'theta_values': episode_states1,
    'num_steps': len(episode_states)
 })
```
