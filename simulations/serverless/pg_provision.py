import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from gym.envs.serverless.faas_params import EnvParameters
from logger import Logger
from ploter import Ploter
from workflow_generator import WorkflowGenerator
from pg_agent import PGAgent


# Set up logger
logger_wrapper = Logger("pg_provision")
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

# Set up policy gradient agent
pg_agent = PGAgent(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dims=[50, 20],
    learning_rate=0.008,
    discount_factor=1
    )

max_episode = 500
reward_trend = []
avg_slow_down_trend = []
timeout_num_trend = []

# Start random provision
for episode in range(max_episode):
    observation = env.reset()
    actual_time = 0
    system_time = 0
    reward_sum = 0
    
    while True:
        actual_time = actual_time + 1
        action = pg_agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        pg_agent.record_trajectory(observation, action, reward)
        
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
            logger.info("Total reward: {}".format(reward_sum))
            
            value = pg_agent.propagate()
            
            reward_trend.append(reward_sum)
            avg_slow_down_trend.append(info["avg_slow_down"])
            timeout_num_trend.append(info["timeout_num"])
            
            break
        
        observation = next_observation

# Plot each episode 
ploter = Ploter()
# ploter.plot_save("PG", reward_trend, avg_slow_down_trend, timeout_num_trend)
ploter.plot_show("PG", reward_trend, avg_slow_down_trend, timeout_num_trend)

