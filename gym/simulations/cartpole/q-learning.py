import gym
import numpy as np

env = gym.make("CartPole-v0")

# Rules
max_number_of_steps = 200
goal_average_steps = 195
num_consecutive_iterations = 100
num_episodes = 2000

# Score stack of recent 100 games
last_time_steps = np.zeros(num_consecutive_iterations)  

# Record 4^4 = 256 states and their corresponding q values
q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))

# Linspace 
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# Transfer observation to a discrete state value between 0 and 255
def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    
    # Linespace each aspect of current state 
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    
    # Transfer to a value between 0 and 255
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

# Get next best action based on current feedback
def get_action(state, action, observation, reward, episode):
    # Digitize next state
    next_state = digitize_state(observation)    
    
    # Decay epsilon and get next action
    epsilon = 0.5 * (0.99 ** episode)
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    
    # Learning rate and discounted factor
    alpha, gamma = 0.2, 0.99 
    
    # Update Q Table
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])
    
    return next_state, next_action

# Start training
stop_flag = False

for episode in range(num_episodes):
    observation = env.reset()   
    state = digitize_state(observation)     
    action = np.argmax(q_table[state])     
    episode_reward = 0
    
    # Start one game
    for t in range(max_number_of_steps):
        env.render()   
        observation, reward, done, info = env.step(action)  
        
        # Penalize agent if done with number of step less than 195
        if done and t < goal_average_steps:
            reward = -max_number_of_steps
        
        # Get next action based on q_table
        state, action = get_action(state, action, observation, reward, episode) 
        episode_reward += reward
        
        # Show the mean of recent 100 games
        if done:
            # Update the stack of recent 100 games
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))  
            print("{} episode: done with {} timesteps, recent mean is {}".format(episode, t + 1, last_time_steps.mean()))
            
            # Set stop_flag to be true if mean score of recent 100 games is over 195
            if (last_time_steps.mean() >= goal_average_steps):
                print("Episode {} succeeded!".format(episode))
                stop_flag = True
            
            break
    
    # Stop if agent learns how to play the game
    if stop_flag is True:
        break

if stop_flag is False:
    print('Failed!')

