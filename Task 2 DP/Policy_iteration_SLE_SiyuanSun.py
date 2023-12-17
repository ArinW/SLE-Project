# Subject: SLE Final Project Q2.1(policy Iteration)

# Author: Sun Siyuan
# Date: 12/15/2023
# Disclaimer: 
# Running this code would satisfy the requirement listed in instruction for "...
# Since there are 4 state variables, generate plots of value function with respect to θ and x for selected ˙θ and ˙x.... 
# At least three plots are needed. Plots of optimal policy, in the same way as in the plots of value function". 
# See README.m for plotting details. 

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# -Physical Constants--------------------------------------------------------------------------------------------------------------------------------------
M = 1.0  # cart mass
m = 0.1  # pole mass
l = 0.5  # length
mu_p = 0.000002 
mu_c = 0.0005 
g = -9.8  # acceleration due to gravity
dt = 0.02 #per requirement
MAX_ITERATIONS = 1000  # prevent infinite loops
CONVERGENCE_THRESHOLD = 1e-4  # Threshold to determine value function convergence
# -Constants for Boxes System
xinfin_neg=-100 # we use ±100 represent extreme Cart Velocity beyond the ±0.5 m/s thresholds
xinfin_pos= 100
thetainfin_neg=-625  #  we use ±500 represent extreme angular velocities beyond the ±50 degrees/s thresholds
thetainfin_pos= 625

# Discretization thresholds from the Boxes system-------------------------------------------------------------------------------------------------
theta_bins = np.array([-12, -6, -1, 0, 1, 6, 12]) * np.pi / 180  # in degrees
x_bins = [-2.4, -0.8, 0.8, 2.4]  # in meters
theta_dot_bins = np.array([-np.inf, -50, 50, np.inf]) * np.pi / 180  # in degrees per second
x_dot_bins = [-np.inf, -0.5, 0.5, np.inf]  # in meters per second

actions = [-10, 10]
num_states = 6334 #arbitraily defined 

def find_region(region_index): #updated
    # Define the ranges for each state variable
    theta_ranges = [(-12, -6), (-6, -1), (-1, 0), (0, 1), (1, 6), (6, 12)]
    x_ranges = [(-2.4, -0.8), (-0.8, 0.8), (0.8, 2.4)]
    theta_dot_ranges = [(-1200, -50), (-50, 50), (50, 1200)]
    x_dot_ranges = [(-240, -0.5), (-0.5, 0.5), (0.5, 240)]

    # Calculate indices for each state variable
    indices = [int(digit) for digit in f"{region_index:04}"]
    
    # Convert degrees to radians for theta and theta_dot ranges
    theta_range = tuple(np.radians(deg) for deg in theta_ranges[indices[0] - 1])
    theta_dot_range = tuple(np.radians(deg) for deg in theta_dot_ranges[indices[2] - 1])

    # Get the range for x and x_dot
    x_range = x_ranges[indices[1] - 1]
    x_dot_range = x_dot_ranges[indices[3] - 1]

    return theta_range, x_range, theta_dot_range, x_dot_range

def rand_p(region_index, Number_of_points):#updated
    #for a given region_index, it generates Number_of_points random points, 
    #where each point is a state vector [theta, x, theta_dot, x_dot] with each component sampled uniformly from its corresponding range. 
    # Get the range for each state variable from the region index
    theta_range, x_range, theta_dot_range, x_dot_range = find_region(region_index)
    
    # Generate a random sample within each range for the specified number of points
    random_points = np.array([[random.uniform(*theta_range), 
                               random.uniform(*x_range), 
                               random.uniform(*theta_dot_range), 
                               random.uniform(*x_dot_range)] 
                              for _ in range(Number_of_points)])
    return random_points   

def discretize_state(theta, x, theta_dot, x_dot): #updated and checked 
    # This discretization allows for a multidimensional state space to be represented as a flat array, 
    # which can be very convenient for indexing and storage,especially in tabular reinforcement learning methods like dynamic programming
    state_index = (np.digitize(theta, theta_bins) * 1000 +
                   np.digitize(x, x_bins) * 100 +
                   np.digitize(theta_dot, theta_dot_bins) * 10 +
                   np.digitize(x_dot, x_dot_bins))
    return state_index

def dynamics(theta, x, theta_dot, x_dot, F):#updated and checked 
    # Pre-calculate common terms
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    mass_term = M + m
    ml_theta_dot_squared_sin_theta = m * l * theta_dot**2 * sin_theta
    mu_c_sign_x_dot = mu_c * np.sign(x_dot)

    # Calculate theta_double_dot
    numerator_theta_ddot = (g * sin_theta + cos_theta * (-F - ml_theta_dot_squared_sin_theta + mu_c_sign_x_dot) / mass_term - mu_p * theta_dot / (m * l))
    denominator_theta_ddot = l * (4.0 / 3.0 - m * cos_theta**2 / mass_term)
    theta_ddot = numerator_theta_ddot / denominator_theta_ddot

    # Calculate x_double_dot
    numerator_x_ddot = F + ml_theta_dot_squared_sin_theta - theta_ddot * m * l * cos_theta - mu_c_sign_x_dot
    x_ddot = numerator_x_ddot / mass_term

    return theta_ddot, x_ddot

def update_state(theta, x, theta_dot, x_dot, F, dt): #updated 
    theta_ddot, x_ddot = dynamics(theta, x, theta_dot, x_dot, F)
    return (theta + theta_dot * dt, 
            x + x_dot * dt, 
            theta_dot + theta_ddot * dt, 
            x_dot + x_ddot * dt)

def is_terminal(theta, x):# determine whether a given state of the cart-pole system is considered a terminal state.
    #State where pole has fallen over beyond a recoverable angle, or the cart has moved too far from the center
    return theta < -12*np.pi/180 or theta > 12*np.pi/180 or x < -2.4 or x > 2.4 #Condition per requirements

def transition_probability(random_points, dt):# Using Bellman Equation
   # Compute p(s'|s,a), the transition probabilities for the cart-pole system 
   # if given a set of random points from the state space and a set of possible actions. Initialize the probability matrix with zeros
    pb = np.zeros((6334, len(actions))) # 6334: number of state, 2: number of action,s.t:(+10 , -10)

    # Loop through each random point and action to update the matrix
    for point in random_points:
        theta, x, theta_dot, x_dot = point

        for a, action in enumerate(actions):
            # Update the state based on the action
            new_theta, new_x, new_theta_dot, new_x_dot = update_state(theta, x, theta_dot, x_dot, action, dt)

            # Determine the index for the new state
            new_index = 0 if is_terminal(new_theta, new_x) else discretize_state(new_theta, new_x, new_theta_dot, new_x_dot)

            # Increment the probability count for the state-action pair
            pb[new_index, a] += 1

    # Normalize the counts to get probabilities
    pb /= len(random_points)

    return pb
       
policy = np.random.choice(actions, size=num_states) # initializes a policy array with random actions
value_function = np.zeros(num_states) # creates an array to hold the value function, initialized to zeros.
# ^This array will be used to store the value (expected cumulative reward) of each state under the policy we customized.

trans_prob = np.zeros((num_states, num_states, 2)) #trans_prob='transitional Probability' 
# This matrix will hold the probabilities of transitioning from one state to another given a particular action. 
# The dimensions correspond to (current state, next state, action).

def decode_state_index(s): #checked 
    theta_index = int(s / 1000)
    x_index = int((s % 1000) / 100)
    theta_dot_index = int((s % 100) / 10)
    x_dot_index = s % 10
    return theta_index, x_index, theta_dot_index, x_dot_index

def is_valid_state(theta_index, x_index, theta_dot_index, x_dot_index): #checked
    return (1 <= theta_index <= 6 and
            1 <= x_index <= 3 and
            1 <= theta_dot_index <= 3 and
            1 <= x_dot_index <= 3)

for s in range(num_states): #updated and checked loop. Get Translational Probability Matrix in this round. 
    # Decode the state index 's' into discretized state variables
    theta_index, x_index, theta_dot_index, x_dot_index = decode_state_index(s)
    # Check if the state index corresponds to a valid state
    if is_valid_state(theta_index, x_index, theta_dot_index, x_dot_index):
        # Generate random points for this state index
        rp = rand_p(s, 100)
        # Calculate the transition probabilities for these points
        trans_prob[s, :, :] = transition_probability(rp, dt)

#------------------Policy iteration loop------------------------
# Maximum valid index values
MAX_THETA_INDEX = 6
MAX_X_INDEX = 3
MAX_THETA_DOT_INDEX = 3
MAX_X_DOT_INDEX = 3
is_policy_stable = False
iterations = 0
iteration2 = 0
#iteration3 = 0
while not is_policy_stable:# needs to embellish n make it more concise. 
    iterations += 1
    print('iteration1=',iterations)
    # Policy Evaluation
    while True:
        iteration2 += 1
        print('iteration2 = ', iteration2)
        delta = 0
        for s in range(num_states):
         #Decompose the state index into individual indices for each state component
            theta_index = int(s / 1000)
            x_index = int((s % 1000) / 100)
            theta_dot_index = int((s % 100) / 10)
            x_dot_index = s % 10

            # Skip invalid state combinations
            if not (1 <= theta_index <= MAX_THETA_INDEX and
                1 <= x_index <= MAX_X_INDEX and
                1 <= theta_dot_index <= MAX_THETA_DOT_INDEX and
                1 <= x_dot_index <= MAX_X_DOT_INDEX):
               continue
            # Update the value function
            current_value = value_function[s]
            action_index = actions.index(policy[s])  # Action for the current policy
            expected_return = sum(trans_prob[s, next_state, action_index] * (1 + value_function[next_state])
                          for next_state in range(1, num_states))
            delta = max(delta, abs(current_value - expected_return))
            value_function[s] = expected_return
            
        if delta < 1e-4:  
            break       
    print('after iteration') 
            
    # Policy Improvement
    is_policy_stable = True
    for s in range(num_states):
        # Decompose the state index into individual components
        theta_index = int(np.floor(s / 1000))
        x_index = int(np.floor((s % 1000) / 100))
        theta_dot_index = int(np.floor((s % 100) / 10))
        x_dot_index = s % 10

        # Skip invalid state combinations
        if not (1 <= theta_index <= 6 and 1 <= x_index <= 3 and 
                1 <= theta_dot_index <= 3 and 1 <= x_dot_index <= 3):
            continue

        # Calculate the expected return for each action and update the policy
        old_action = policy[s]
        expected_returns = [sum(trans_prob[s, next_state, action] * (1 + value_function[next_state])
                                for next_state in range(1, num_states)) for action in range(len(actions))]

        # Update the policy with the action leading to the highest expected return
        best_action_index = np.argmax(expected_returns)
        policy[s] = actions[best_action_index]

        # Check if the policy has changed
        if old_action != policy[s]:
            is_policy_stable = False
    print('after policy improvement')
    
    if iterations > 1000:  # prevent infinite loops
        print("Stopping due to too many iterations")
        break

def specify_point(region_index):#Checked and proofed. 
    # Constants for conversion
    DEG_TO_RAD = np.pi / 180
    THETA_VALUES = [-12, -3.5, -0.5, 0.5, 3.5, 12]# Angle is discretized into bins of 0, ±1, ±6, and ±12 degrees
    THETA_DOT_VALUES = [thetainfin_neg, 0, thetainfin_pos] # Calling Constants from Previously Defined Numbers.             
    X_VALUES = [-2.4, 0.0, 2.4]                   # The thresholds for the cart position are ±0.8 and ±2.4 meters
    X_DOT_VALUES = [xinfin_neg, 0, xinfin_pos]
    
    # Calculating indices
    theta_index = int(np.floor(region_index / 1000))
    x_index = int(np.floor((region_index % 1000) / 100))
    theta_dot_index = int(np.floor((region_index % 100) / 10))
    x_dot_index = region_index % 10
    
    # Mapping to values
    theta = THETA_VALUES[theta_index - 1] * DEG_TO_RAD
    x = X_VALUES[x_index - 1]
    theta_dot = THETA_DOT_VALUES[theta_dot_index - 1] * DEG_TO_RAD
    x_dot = X_DOT_VALUES[x_dot_index - 1]

    return theta, x, theta_dot, x_dot

# Create "results" matrices: 
# 2D NumPy array where each row represents a different state, with columns containing the discretized values for 
# theta (pole angle), x (cart position), theta_dot (angular velocity of the pole), x_dot (velocity of the cart),
# and the corresponding value from the value function.
results = []
i = 0
results = np.zeros((162,5)) # 162 tabulated region x 5 columns (4 states + 1 value)
for theta_index in range(1, 7):  # Assuming theta_index goes from 1 to 6 inclusive
    for x_index in range(1, 4):  # Assuming x_index goes from 1 to 3 inclusive
        for theta_dot_index in range(1, 4):  # Assuming theta_dot_index goes from 1 to 3 inclusive
            for x_dot_index in range(1, 4):  # Assuming x_dot_index goes from 1 to 3 inclusive
                # Reconstruct the state index 's' from the valid indices
                s = (theta_index * 1000) + (x_index * 100) + (theta_dot_index * 10) + x_dot_index
                
                # Now 's' is guaranteed to be in the valid range, so we can skip the 'if' check
                point_index = specify_point(s)
                point_array = np.array(point_index)
                point_array = np.append(point_array, value_function[s])
                
                # Store the results
                results[i, :] = point_array
                i += 1

#---End of Policy Itertation----------------------------------------------------------------------------------------------------------------
#---Plotting Session-------------------------------------------------------------------------------------------------------------------------  
# results[:,2]--->accesses all rows and the third column, which is our discretized theta_dot values for each state
# results[:,3]--->accesses all rows and the fourth column, which is our discretized x_dot values for each state. 
discre_theta_dot= results[:,2]
discre_x_dot    = results[:,3]

#*********README: 
# To get plot, comment out unwanted secnarios for 'filtered results" and make sure select corresponding title.  
# 3 cases are being proposed for illustration of the plot of value function vs theta and x under selected theta_dot and x_dot. 

#1st case, we will use Central bins for both θ_dot and x_dot.
#filtered_results = results[(discre_theta_dot == 0*np.pi/180) & (discre_x_dot == 0)]

#2nd case: Lowest bin for θ_dot and central bin for x_dot.
#filtered_results = results[(discre_theta_dot == thetainfin_neg*np.pi/180) & (discre_x_dot == 0)]

#3rd case: Highest bin for θ_dot and central bin for x_dot
filtered_results = results[(discre_theta_dot == thetainfin_pos*np.pi/180) & (discre_x_dot == 0)]

x = filtered_results[:,0]*180/np.pi # converted in deg 
y = filtered_results[:,1]
z = filtered_results[:,4]

xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x, y), z, (xi, yi), method='cubic') 
# The griddata function is used to interpolate the value function over a grid defined by xi and yi. This is 
# necessary because the value function may not be available at every point in the space, 
# so griddata fills in the gaps to create a smooth surface:

fig = plt.figure(figsize=(20, 10),dpi=300)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis')
cbar = fig.colorbar(surf, shrink=0.7, aspect=21) 

labelsize = 6  #choose the size that fits best
theta_dot_value_high=  50
theta_dot_value_mid=    0
theta_dot_value_low = -50  
x_dot_value = 0  # Example value

# title for case 1: Central bins for both θ_dot and x_dot
#title = r'Value Function vs $\theta$ and $x$ for $\dot{\theta}='+f'{theta_dot_value_mid}' + r'^\circ/s$, and $\dot{x}=' + f'{x_dot_value}' + r'm/s$'
# title for case 2: Lowest bin for θ_dot and central bin for x_dot.
#title = r'Value Function vs $\theta$ and $x$ for $\dot{\theta}<'+f'{theta_dot_value_low}' + r'^\circ/s$, and $\dot{x}=' + f'{x_dot_value}' + r'm/s$'
# title for case 3: Highest bin for θ_dot and central bin for x_dot
title = r'Value Function vs $\theta$ and $x$ for $\dot{\theta}>'+f'{theta_dot_value_high}' + r'^\circ/s$, and $\dot{x}=' + f'{x_dot_value}' + r'm/s$'

# Set the title to the plot with a specific fontsize
ax.set_title(title, fontsize=8)
ax.set_xlabel(r'Pole Angle $\theta$ (degrees)', fontsize=labelsize)
ax.set_ylabel('Cart Position x (meters)', fontsize=labelsize)
ax.set_zlabel('Value Function', fontsize=labelsize)
ax.tick_params(axis='x', labelsize=7) # adjust size to increase readability. 
ax.tick_params(axis='y', labelsize=7)
ax.tick_params(axis='z', labelsize=7)
plt.tight_layout()
plt.show()

i = 0
policy_results = np.zeros((162, 5))
for s in range(num_states):
    theta_index = s // 1000
    x_index = (s % 1000) // 100
    theta_dot_index = (s % 100) // 10
    x_dot_index = s % 10

    # Check if the state is within the specified ranges
    if not (1 <= theta_index <= 6 and 1 <= x_index <= 3 and 
            1 <= theta_dot_index <= 3 and 1 <= x_dot_index <= 3):
        continue

    # Store the current state's data
    policy_results[i] = [theta_index, x_index, theta_dot_index, x_dot_index, policy[s]]
    i += 1

# Filter and retain rows where both theta_dot_index and x_dot_index are 3
#filtered_results_p = policy_results[(policy_results[:, 2] == 2) & (policy_results[:, 3] == 2)] #case1
#filtered_results_p = policy_results[(policy_results[:, 2] == 1) & (policy_results[:, 3] == 2)] #case2
filtered_results_p = policy_results[(policy_results[:, 2] == 3) & (policy_results[:, 3] == 2)] #case3

optimal_policy = np.zeros((3, 6))
# Iterate directly over rows of filtered_results_p
for row in filtered_results_p:
    # Subtract 1 from the indices to convert from 1-based to 0-based indexing
    x_index = int(row[1] - 1)
    y_index = int(row[0] - 1)
  # Update the optimal policy matrix
    optimal_policy[x_index, y_index] = row[4]

print('Optimal policy:')
print(optimal_policy)

# Assuming 'x' and 'y' are predefined arrays or lists
xi = np.linspace(min(x), max(x), 2)
yi = np.linspace(min(y), max(y), 2)
xi, yi = np.meshgrid(xi, yi)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Iterate over filtered_results to plot arrows
for i in range(len(filtered_results)):
    x_pos = filtered_results[i, 0] * 180 / np.pi # Convert from radians to degrees
    y_pos = filtered_results[i, 1]
    # Determine the direction of the arrow
    x_direct = -1 if filtered_results_p[i, 4] == -10 else 1
    y_direct = 0
    # Add the arrow to the plot
    ax.quiver(x_pos, y_pos, x_direct, y_direct, angles='xy', scale_units='xy', scale=2, color='red')

# Set labels and title
plt.xlabel('Pole Angle θ (degrees)',fontsize=5)
plt.ylabel('Position of Moving Cart x (meters)',fontsize=5)
#plt.title(r'Optimal Policy under Value Iteration with ($\dot{\theta}$=0.0°/s, $\dot{x}$=0.0m/s) (Case 1)',fontsize=4) #case 1
#plt.title(r'Optimal Policy under Value Iteration with ($\dot{\theta}$<50°/s, $\dot{x}$=0.0m/s) (case 2)',fontsize=4) #case 2
plt.title(r'Optimal Policy under Value Iteration with ($\dot{\theta}$>50°/s, $\dot{x}$=0.0m/s) (case 3)',fontsize=4) #case 3
plt.show()
