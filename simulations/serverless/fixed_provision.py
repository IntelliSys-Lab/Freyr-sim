import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter



#
# Fixed provision strategy
#
def fixed_provision(
    profile,
    timetable,
    env_params,
    max_episode=500,
    plot_prefix_name="Fixed",
    save_plot=False,
    show_plot=True,
):
    # Set up logger
    logger_wrapper = Logger("fixed_provision")
    logger = logger_wrapper.get_logger()
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    reward_trend = []
    avg_slow_down_trend = []
    timeout_num_trend = []
    
    # Start random provision
    for episode in range(max_episode):
        observation = env.reset()
        reward_sum = 0
        actual_time = 0
        system_time = 0
        
        action = env.action_space.n - 1
        while True:
            actual_time = actual_time + 1
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
                avg_slow_down = info["avg_slow_down"]
                timeout_num = info["timeout_num"]
                
                logger.info("")
                logger.info("**********")
                logger.info("**********")
                logger.info("**********")
                logger.info("")
                logger.info("Episode {} finished after:".format(episode))
                logger.info("{} actual timesteps".format(actual_time))
                logger.info("{} system timesteps".format(system_time))
                logger.info("Total reward: {}".format(reward_sum))
                logger.info("Avg slow down: {}".format(avg_slow_down))
                logger.info("Timeout num: {}".format(timeout_num))
                
                reward_trend.append(reward_sum)
                avg_slow_down_trend.append(avg_slow_down)
                timeout_num_trend.append(timeout_num)
                
                break
    
    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(plot_prefix_name, reward_trend, avg_slow_down_trend, timeout_num_trend)
    if show_plot is True:
        plotter.plot_show(reward_trend, avg_slow_down_trend, timeout_num_trend)

    logger_wrapper.shutdown_logger()
    
    