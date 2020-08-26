import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from gym.envs.serverless.faas_params import EnvParameters
from logger import Logger
from ploter import Ploter
from workflow_generator import WorkflowGenerator


# Set up logger
logger_wrapper = Logger("random_provision")
logger = logger_wrapper.get_logger()

# Generate workflow
workflow_generator = WorkflowGenerator()
profile, timetable = workflow_generator.generate_workflow()

# Set paramters for FaaSEnv
env_params = EnvParameters(
    cpu_total=32*10,
    memory_total=46*10,
    cpu_cap_per_function=32,
    memory_cap_per_function=46
    )

# Make environment
env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
env.seed(114514) # Reproducible, policy gradient has high variance

max_episode = 500
reward_trend = []
avg_slow_down_trend = []
timeout_num_trend = []

# Start random provision
for episode in range(max_episode):
    observation = env.reset()
    reward_sum = 0
    actual_time = 0
    system_time = 0
    
    while True:
        actual_time = actual_time + 1
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if system_time < info["system_time"]:
            system_time = info["system_time"]
            
        logger.debug("")
        logger.debug("Actual timestep {}".format(actual_time))
        logger.debug("System timestep {}".format(system_time))
        logger.debug("Take action: {}".format(action))
        logger.debug("Observation: {}".format(observation))
        logger.debug("Reward: {}".format(reward))
        
        reward_sum = reward_sum + reward
        
        if done:
            logger.info("")
            logger.info("**********")
            logger.info("**********")
            logger.info("**********")
            logger.info("")
            logger.info("Episode {} finished after:".format(episode))
            logger.info("{} actual timesteps".format(actual_time))
            logger.info("{} system timesteps".format(system_time))
            logger.info("total reward is {}".format(reward_sum))
            
            reward_trend.append(reward_sum)
            avg_slow_down_trend.append(info["avg_slow_down"])
            timeout_num_trend.append(info["timeout_num"])
            
            break

# Plot each episode 
ploter = Ploter()
ploter.plot_save("Random", reward_trend, avg_slow_down_trend, timeout_num_trend)
ploter.plot_show("Random", reward_trend, avg_slow_down_trend, timeout_num_trend)

