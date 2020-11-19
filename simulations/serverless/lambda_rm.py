import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter
from pg_ppo2_agent import PPO2Agent



#
# Policy gradient provision strategy
#

def lambda_rm_train(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=200,
    model_save_path="ckpt/best_model.pth",
    save_plot=False,
    show_plot=True,
):
    rm = "LambdaRM_train"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    # Set up policy gradient agent
    pg_agent = PPO2Agent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=[64, 32],
        learning_rate=0.002,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_steps=5
    )
    
    # Trends recording
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    loss_trend = []

    # Pinpoint best avg completion time model
    min_avg_completion_time = 10e8
    
    # Start random provision
    for episode in range(max_episode):
        observation = env.reset()
        pg_agent.reset()

        actual_time = 0
        system_time = 0
        reward_sum = 0
        
        while True:
            actual_time = actual_time + 1
            action, value_pred, log_prob = pg_agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)

            pg_agent.record_trajectory(
                observation=observation, 
                action=action, 
                reward=reward,
                value=value_pred,
                log_prob=log_prob
            )
            
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
                loss = pg_agent.propagate()
                avg_completion_time = info["avg_completion_time"]
                timeout_num = info["timeout_num"]

                # Save best model that has min avg completion time
                if avg_completion_time < min_avg_completion_time:
                    min_avg_completion_time = avg_completion_time
                    pg_agent.save(model_save_path)
                
                logger.info("")
                logger.info("**********")
                logger.info("**********")
                logger.info("**********")
                logger.info("")
                logger.info("Episode {} finished after:".format(episode))
                logger.info("{} actual timesteps".format(actual_time))
                logger.info("{} system timesteps".format(system_time))
                logger.info("Total reward: {}".format(reward_sum))
                logger.info("Avg completion time: {}".format(avg_completion_time))
                logger.info("Timeout num: {}".format(timeout_num))
                logger.info("Loss: {}".format(loss))
                
                reward_trend.append(reward_sum)
                avg_completion_time_trend.append(avg_completion_time)
                timeout_num_trend.append(timeout_num)
                loss_trend.append(loss)
                
                break
            
            observation = next_observation
    
    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(
            prefix_name=rm, 
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend, 
            loss_trend=loss_trend
        )
    if show_plot is True:
        plotter.plot_show(
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend, 
            timeout_num_trend=timeout_num_trend, 
            loss_trend=loss_trend
        )

    
def lambda_rm_eval(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=10,
    checkpoint_path="ckpt/best_model.pth",
    save_plot=False,
    show_plot=True,
):
    rm = "LambdaRM_eval"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    # Set up policy gradient agent
    pg_agent = PPO2Agent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=[64, 32],
        learning_rate=0.002,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_steps=5
    )

    # Restore checkpoint model
    pg_agent.load(checkpoint_path)
    
    # Trends recording
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    loss_trend = []
    
    # Start random provision
    for episode in range(max_episode):
        observation = env.reset()
        pg_agent.reset()

        actual_time = 0
        system_time = 0
        reward_sum = 0
        
        while True:
            actual_time = actual_time + 1
            action, value_pred, log_prob = pg_agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)

            pg_agent.record_trajectory(
                observation=observation, 
                action=action, 
                reward=reward,
                value=value_pred,
                log_prob=log_prob
            )
            
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
                loss = pg_agent.propagate()
                avg_completion_time = info["avg_completion_time"]
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
                logger.info("Avg completion time: {}".format(avg_completion_time))
                logger.info("Timeout num: {}".format(timeout_num))
                logger.info("Loss: {}".format(loss))
                
                reward_trend.append(reward_sum)
                avg_completion_time_trend.append(avg_completion_time)
                timeout_num_trend.append(timeout_num)
                loss_trend.append(loss)
                
                break
            
            observation = next_observation
    
    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(
            prefix_name=rm, 
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend, 
            loss_trend=loss_trend
        )
    if show_plot is True:
        plotter.plot_show(
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend, 
            timeout_num_trend=timeout_num_trend, 
            loss_trend=loss_trend
        )

    