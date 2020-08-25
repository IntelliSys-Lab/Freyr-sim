import sys
sys.path.append("../../gym")
from logger import get_logger
import gym
from gym.envs.serverless.faas_utils import Function, Application, Profile, Timetable
from gym.envs.serverless.faas_params import EnvParameters, FunctionParameters


# Set up logger
logger = get_logger("random_provision")

# Create 4 functions, initially over-provision all the functions
function_params_1 = FunctionParameters(
    ideal_cpu=4, 
    ideal_memory=4, 
    ideal_duration=1, 
    cpu_least_hint=1, 
    memory_least_hint=1,
    timeout=60,
    cpu_cap_per_function=32,
    memory_cap_per_function=46
    )
function_1 = Function(function_params_1)
function_1.set_function(cpu=32, memory=46)

function_params_2 = FunctionParameters(
    ideal_cpu=16, 
    ideal_memory=16, 
    ideal_duration=5, 
    cpu_least_hint=8, 
    memory_least_hint=8,
    timeout=60,
    cpu_cap_per_function=32,
    memory_cap_per_function=46,
    )
function_2 = Function(function_params_2)
function_2.set_function(cpu=32, memory=46)

function_params_3 = FunctionParameters(
    ideal_cpu=24, 
    ideal_memory=32, 
    ideal_duration=30, 
    cpu_least_hint=12, 
    memory_least_hint=16,
    timeout=60,
    cpu_cap_per_function=32,
    memory_cap_per_function=46,
    )
function_3 = Function(function_params_3)
function_3.set_function(cpu=32, memory=46)

function_params_4 = FunctionParameters(
    ideal_cpu=32, 
    ideal_memory=46, 
    ideal_duration=50, 
    cpu_least_hint=16, 
    memory_least_hint=23,
    timeout=60,
    cpu_cap_per_function=32,
    memory_cap_per_function=46,
    )
function_4 = Function(function_params_4)
function_4.set_function(cpu=32, memory=46)

application_1 = Application([function_1, function_2])
application_2 = Application([function_3, function_4])

function_list = [function_1, function_2, function_3, function_4]
application_list = [application_1, application_2]
p = Profile(application_list, function_list)

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

t = Timetable(timetable)

# Set paramters for FaaSEnv
env_params = EnvParameters(
    cpu_total=32*10,
    memory_total=46*10,
    cpu_cap_per_function=32,
    memory_cap_per_function=46
    )


# Make environment
env = gym.make("FaaS-v0", params=env_params, profile=p, timetable=t)

episode_num = 1
max_timestep = 2000

# Start random provision
for episode in range(episode_num):
    observation = env.reset()
    reward_total = 0
    system_time = 0
    
    for t in range(max_timestep):
        action = None
        observation, reward, done, info = env.step(action)
        
        if system_time < info["system_time"]:
            system_time = info["system_time"]
            logger.info("")
            logger.info("Actual timestep {}".format(t+1))
            logger.info("System timestep {}".format(system_time))
            logger.info("Take action: {}".format(action))
            logger.info("Observation: {}".format(observation))
            logger.info("Reward: {}".format(reward))
        else:
            logger.debug("")
            logger.debug("Actual timestep {}".format(t+1))
            logger.debug("System timestep {}".format(system_time))
            logger.debug("Take action: {}".format(action))
            logger.debug("Observation: {}".format(observation))
            logger.debug("Reward: {}".format(reward))
        
        reward_total = reward_total + reward
        
        if done:
            logger.info("")
            logger.info("**********")
            logger.info("**********")
            logger.info("**********")
            logger.info("")
            logger.info("Episode finished after:")
            logger.info("{} actual timesteps".format(t+1))
            logger.info("{} system timesteps".format(system_time))
            logger.info("total reward is {}".format(reward_total))
            break

