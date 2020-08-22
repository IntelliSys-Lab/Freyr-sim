import sys
sys.path.append("../../gym")
import gym
from gym.envs.serverless.faas_utils import Function, Application

# Create 4 functions, initially over-provision all the functions
function_1 = Function(
    ideal_cpu=1, 
    ideal_memory=4, 
    ideal_duration=1, 
    cpu_least_hint=1, 
    memory_least_hint=1,
    timeout=60
    )
function_1.set_function(cpu=2, memory=46)

function_2 = Function(
    ideal_cpu=1, 
    ideal_memory=16, 
    ideal_duration=5, 
    cpu_least_hint=1, 
    memory_least_hint=8,
    timeout=60
    )
function_2.set_function(cpu=2, memory=46)

function_3 = Function(
    ideal_cpu=2, 
    ideal_memory=32, 
    ideal_duration=30, 
    cpu_least_hint=1, 
    memory_least_hint=16,
    timeout=60
    )
function_3.set_function(cpu=2, memory=46)

function_4 = Function(
    ideal_cpu=2, 
    ideal_memory=46, 
    ideal_duration=50, 
    cpu_least_hint=1, 
    memory_least_hint=24,
    timeout=60
    )
function_4.set_function(cpu=2, memory=46)

app_1 = Application([function_1, function_2])
app_2 = Application([function_3, function_4])

repo = [function_1, function_2, function_3, function_4]

# Set up timetable
timetable = []
timesteps = 120

for i in range(timesteps):
    time = []
    
    if i%1 == 0:
        time.append(function_1.function_id)
    if i%2 == 0:
        time.append(function_2.function_id)
    if i%8 == 0:
        time.append(function_3.function_id)
    if i%10 == 0:
        time.append(function_4.function_id)
        
    timetable.append(time)


# Make environment
env = gym.make("FaaS-v0", repo=repo, timetable=timetable, total_cpu=2*10, total_memory=46*10)

episode_num = 1
max_timestep = 2000

# Start random provision
for episode in range(episode_num):
    observation = env.reset()
    
    for t in range(max_timestep):
        print("Timestep {}:".format(t))
        
        action = env.action_space.sample()
        print("Take action: {}".format(action))
        
        observation, reward, done, info = env.step(action)
        print("Observation: {}".format(observation))
        print("Reward: {}".format(reward))
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

