"""
Environment module

Copied and modified from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
Modification needed for discretisation of the state
"""

from math import cos, sin, pi
import numpy as np

class CartPoleEnv():
    def __init__(self):

        ##SETTINGS
        #Settings for mechanical modelisation
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = .5        #(half the size)
        self.force_mag = 10.0
        #Setting for numerical calcul
        self.tau = 0.02         #setting for Euler's method
        self.dt_system = 0.02    #reaction time of the systeme (for real application)
        #Settings for borders
        self.x_limit = 1                         #maximum position
        self.theta_limit = 40 * 2 * pi / 360   #maximum angle
        self.x_dot_limit = 15                     #maximum linear velocity
        self.theta_dot_limit = 15                 #maximum angular velocity
        #Others settings
        self.nb_discretisation = 11           #number of discretisation for each of the 4 states
        self.Recompenses = [-10000, 0, 0, 100]   #reward for [near to border, neutral, vertical pole, vertical pole and cart in the center]


        ##INITIALIZATION
        #Declaration of intermediate variable
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        #Declaration of others variables
        done = False  #boolean for the end of and epoch
        self.nb_discretisation_x = self.nb_discretisation_x_dot = self.nb_discretisation_theta = self.nb_discretisation_theta_dot = self.nb_discretisation
        self.viewer = None
        self.state = None
        #Declaration of discretization intervals
        self.X = np.linspace(-self.x_limit, self.x_limit, self.nb_discretisation_x)
        self.X_dot = np.linspace(-self.x_dot_limit, self.x_dot_limit, self.nb_discretisation_x_dot)
        self.THETA = np.linspace(-self.theta_limit, self.theta_limit, self.nb_discretisation_theta)
        self.THETA_dot = np.linspace(-self.theta_dot_limit, self.theta_dot_limit, self.nb_discretisation_theta_dot)


    def step(self, action):
        #Mechanical modeling of the inverted pendulum
        for i in range(int(self.dt_system/self.tau)):
            state = self.state
            x, x_dot, theta, theta_dot = state
            force = self.force_mag if action == 1 else -self.force_mag
            costheta = cos(theta)
            sintheta = sin(theta)

            temp = (force - self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (1 - self.masspole * costheta * costheta / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

            self.state = (x, x_dot, theta, theta_dot)
            self.state_discrete = self.continu_to_discrete(self.state)
        pos_cart, vit_cart, pos_pole, vit_pole = self.state_discrete ##

        #Check the end of epoch
        last_index_state = len(self.X)
        mid_index_state = len(self.X) // 2
        if pos_pole == 0 or pos_pole == last_index_state or pos_cart == 0 or pos_cart == last_index_state or vit_pole == 0 or vit_pole == last_index_state or vit_cart == 0 or vit_cart == last_index_state:
            done = True
        else:
            done = False

        #Awarding of rewards
        negative_reward, neutral_reward, positive_reward, outstanding_reward = self.Recompenses

        if (pos_pole in [mid_index_state-1, mid_index_state, mid_index_state+1]) and (pos_cart in [mid_index_state-1, mid_index_state, mid_index_state+1]):
            reward = outstanding_reward
        elif (pos_pole in [mid_index_state-1, mid_index_state, mid_index_state+1]):
            reward = positive_reward
        elif pos_cart == 0 or pos_cart == last_index_state or pos_pole == 0 or pos_pole == last_index_state:
            reward = negative_reward
        else:
            reward = neutral_reward
        return self.state_discrete, reward, done


    def reset(self):
        self.state = [0, 0, 0, 0]
        self.state_discrete = self.continu_to_discrete(self.state)
        return np.array(self.state_discrete)


    def continu_to_discrete(self, state):
        state_discrete = [0, 0, 0, 0]
        x, x_dot, theta, theta_dot = state
        for i in range(len(self.X)-1):
            if self.X[i] <= x < self.X[i+1]:
                state_discrete[0] = i
            if self.X_dot[i] <= x_dot < self.X_dot[i+1]:
                state_discrete[1] = i
            if self.THETA[i] <= theta < self.THETA[i+1]:
                state_discrete[2] = i
            if self.THETA_dot[i] <= theta_dot < self.THETA_dot[i+1]:
                state_discrete[3] = i
        return state_discrete


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_limit*2
        scale = screen_width / world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
