# The Inverted Pendulum Balancing Robot [![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

<p align="center"><img style="display: block; width: 100%; margin: auto;" src="images/header_github.png" /></p>

<p align="justify"><b>The Inverted Pendulum Balancing Robot is an introductory project to reinforcement learning</b> carried out as part of an entrance exam to the "Grandes Ecoles d'Ingénieurs" of the French engineering schools.

The goal of my project was to learn the basics of reinforcement learning.</p>



## Contents:
0. [Installation](#installation)
1. [Intoduction](#introduction)
2. [Approach](#approach)
3. [Results](#results)
4. [Files structure](#file)



<a id='installation'></a>
## Installation

1. Clone this copy to your local disk
```
$ git clone https://github.com/steyerj/inverted-pendulum-balancing-robot.git
$cd inverted-pendulum-balancing-robot
```

2. Install dependecies
```
$ pip install -r requirements.txt
```

2. Run the program
```
$ py main.py
```



<a id='introduction'></a>
## Introduction

<p align="justify">The objective of this project is to teach a robotic cart to balance an inverted pendulum by moving the cart along a linear axis. The agent in this case the cart, will have to be able to understand through the acquired experience the combinations of actions to be performed depending on the state of the pendulum to keep the pole in its unstable equilibrium position.

The objective being to achieve [this type of behavior.](https://www.youtube.com/watch?v=Lt-KLtkDlh8)

This will be achieved using a reinforcement learning algorithm.</p>



<a id='approach'></a>
## Approach

Reinforcement learning can be understood by the following figure:

<p align="center"><img style="display: block; margin: auto; width: 50%" src="images/fig1_rl_schema.png" /><br>
<i class="center">Figure 1 - Explanatory diagram of reinforcement learning</i></p>

<p align="justify">Figure 1 shows a structure in two interacting blocks. This structure is reproduced in the code of this project in the shape of the two .py files in the GitHub directory. <b>An agent</b> (here the robotic cart) interacts with <b>the environment</b> (2D physical model of the behavior of a pendulum) thanks to actions <i>At</i> (direction of a force applied on the cart) then the environment returns a state <i>St</i> (position and speed of the cart and the pendulum pole) and a reward <i>Rt</i> (an integer high or low depending on whether the state reached at time t is close to the desired objective). </p>


### The agent
<p align="justify">The goal of the agent, or more specifically of the algorithm that governs his behavior, is to maximize the value of the rewards obtained. To do so, it records a weighted average of the rewards obtained according to the actions At taken in the environment at the state St.

Among the many reinforcement learning algorithms available in the literature, I chose the <b>Q-Learning</b> algorithm. It has the advantage of being very popular, easy to use and therefore very well documented. It is used in its simplest form, without using deep learning technics to be easier to understand although more limited.

The Q-Learning algorithm works using a matrix called Q-table, which has each doublet of state and action (St,At) associated with the weighted average of the rewards obtained.</p>

<p align="center"><img style="display: block; margin: auto;" src="images/Qtable.png" /><br>
<i>Figure 2 - Example of Qtable</i></p>

<p align="justify">It is by reading and updating this table that the agent creates a representation of the environment allowing him to accomplish his goal. It is nevertheless necessary to discretize the environment in which the agent evolves. This is one of the first limits of the chosen algorithm because, depending on the fineness of the discretization, one can quickly be confronted with the "Curse of dimensionality", the number of discretized states becomes too large to be computed efficiently.</p>

<p align="center"><img style="display: block; margin: auto;" src="images/discretisation_states.jpg" /><br>
<i>Figure 3 - Discretization of the states used</i></p>


### The environment
<p align="justify">It was also necessary to model the environment in which the agent evolves, in our case a two-dimensional physical model of the pendulum's behavior.
It can be obtained simply by applying <i>"Newton's Laws of Motion"</i>.</p>

<p align="center"><img style="display: block; margin: auto;" src="images/pendulum_model.png" /><br>
<i>Figure 4 - Physical model used</i></p>

<p align="justify">
These equations are solved numerically using the <i>"Euler Method"</i>.

In order to be able to visualize the behavior of the cart, a modified version of the "CartPole-v1" environment from the gym library is used.</p>



<a id='results'></a>
## Results

The execution of the program initially gave modest performances: the bar hold was less than 4 seconds on average.
<p align="center"><img style="display: block; margin: auto;" src="images/performance_1.png" /><br>
<i>Figure 5 - Graph showing the average bar hold time as a function of the observed generation</i></p>

<p align="justify">These performances can be improved by a classical technique of reinforcement learning, which consists of implementing code to respond to the exploration/exploitation dilemma. The agent must be able to perform random actions with a decreasing frequency compared to the generation under consideration in order to be able to explore its environment and then, in the end, reach a behavior that only follows the results of the acquired experience.</p>

<p align="center"><img style="display: block; margin: auto;" src="images/performance_2.png" /><br>
<i>Figure 6 - Graph illustrating the decrease in random actions</i></p>

<p align="justify">It can be seen that the performance have doubled from the previous version thanks to the application of this technique.

Finally, the most effective way to increase performance significantly is to increase the rate of discretization. Tripling the discretization rate at the cost of a long computation time will give better results.</p>

<p align="center"><img style="display: block; margin: auto;" src="images/performance_3.png" /><br>
<i>Figure 7 - Graph of final performancee</i></p>



<a id='file'></a>
## File structure

### main .py
The program can be started by simply running 'main.py', but the library <i>numpy</i> must be installed first.
It is possible to modify the hyperparameters of the algorithm by modifying the variables contained in the 'main.py' file.

```python
#Settings for Qlearning's algorithm
learning_rate = 0.6
gamma = 0.99
iteration = 700
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
self.x_limit = 1                         #maximum position
self.theta_limit= 40 * 2 * pi / 360   #maximum angle
self.x_dot_limit = 15                     #maximum linear velocity
self.theta_dot_limit = 15                 #maximum angular velocity
#Others settings
self.nb_discretisation = 11           #number of discretisation for each of the 4 states
self.Recompenses=[-10000, 0, 0, 100]   #reward for [near to border, neutral, vertical pole, vertical pole and cart in the center]
```


### agent.py
The *agent.py* file contains the Q-Learning algorithm which is based on the pseudocode available in *Sutton, R. S., Barto, A. G., Reinforcement Learning: An Introduction [archive]. MIT Press, 1998. 2e édition MIT Press en 2018.*

<p align="center"><img style="display: block; margin: auto;" src="images/QLearning_algorithme.png" /><br>
<i>Figure 8 - Pseudo code of the Q-learning algorithm</i></p>

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
