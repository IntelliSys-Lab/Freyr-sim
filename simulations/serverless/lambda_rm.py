import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter
from ppo2_agent import PPO2Agent
from utils import log_trends, log_resource_utils, log_function_throughput



#
# Policy gradient provision strategy
#

def lambda_rm_train(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=1000,
    hidden_dims_cpu=[64, 32],
    hidden_dims_memory=[64, 32],
    learning_rate=0.001,
    discount_factor=1,
    ppo_clip=0.2,
    ppo_steps=5,
    model_save_path="ckpt/best_model.pth",
    save_plot=False,
    show_plot=True,
):
    rm = "LambdaRM_train"
    function_profile = profile.get_function_profile()

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    # Set up policy gradient agent
    pg_agent = PPO2Agent(
        observation_dim_cpu=env.observation_dim_cpu,
        observation_dim_memory=env.observation_dim_memory,
        action_dim_cpu=env.action_dim_cpu,
        action_dim_memory=env.action_dim_memory,
        hidden_dims_cpu=hidden_dims_cpu,
        hidden_dims_memory=hidden_dims_memory,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        ppo_clip=ppo_clip,
        ppo_steps=ppo_steps
    )
    
    # Trends recording
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    loss_trend = []
    avg_completion_time_per_function_trend = {}
    for function_id in function_profile.keys():
        avg_completion_time_per_function_trend[function_id] = []

    # Record max sum rewards
    max_reward_sum = -10e8
    
    # Start random provision
    for episode in range(max_episode):
        env.reset()
        pg_agent.reset()

        actual_time = 0
        system_time = 0
        reward_sum = 0

        function_throughput_list = []
        
        episode_done = False
        while episode_done is False:
            actual_time = actual_time + 1

            timestep = timetable.get_timestep(system_time)
            if timestep is not None:
                for index, function_id in enumerate(timestep.keys()):
                    observation_cpu, observation_memory = env.get_observation(function_id)

                    if index == len(timestep) - 1:
                        time_proceed = True
                    else:
                        time_proceed = False

                    action_cpu, action_memory, value_pred_cpu, value_pred_memory, log_prob_cpu, log_prob_memory = pg_agent.choose_action(
                        observation_cpu=observation_cpu, 
                        observation_memory=observation_memory
                    )

                    reward, done, info = env.step(
                        time_proceed=time_proceed,
                        function_id=function_id,
                        action_cpu=action_cpu,
                        action_memory=action_memory
                    )

                    pg_agent.record_trajectory(
                        observation_cpu=observation_cpu, 
                        observation_memory=observation_memory, 
                        action_cpu=action_cpu, 
                        action_memory=action_memory, 
                        value_cpu=value_pred_cpu, 
                        value_memory=value_pred_memory, 
                        log_prob_cpu=log_prob_cpu,
                        log_prob_memory=log_prob_memory,
                        reward=reward
                    )
            
                    if system_time < info["system_time"]:
                        system_time = info["system_time"]
                        function_throughput_list.append(info["function_throughput"])
                        
                    logger.debug("")
                    logger.debug("Actual timestep {}".format(actual_time))
                    logger.debug("System timestep {}".format(system_time))
                    logger.debug("Observation cpu: {}".format(observation_cpu))
                    logger.debug("Observation memory: {}".format(observation_memory))
                    logger.debug("Take action cpu: {}".format(action_cpu))
                    logger.debug("Take action memory: {}".format(action_memory))
                    logger.debug("Reward: {}".format(reward))
                    
                    reward_sum = reward_sum + reward
            else:
                time_proceed = True
                function_id = None
                action_cpu = None
                action_memory = None

                reward, done, info = env.step(
                    time_proceed=time_proceed,
                    function_id=function_id,
                    action_cpu=action_cpu,
                    action_memory=action_memory
                )

                reward_sum = reward_sum + reward
            
                if done:
                    loss = pg_agent.learn()
                    avg_completion_time = info["avg_completion_time"]
                    timeout_num = info["timeout_num"]

                    # Save the best model
                    if max_reward_sum < reward_sum:
                        max_reward_sum = reward_sum
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

                    # # Log average completion time per function
                    # request_record = info["request_record"]
                    # for function_id in avg_completion_time_per_function_trend.keys():
                    #     avg_completion_time_per_function_trend[function_id].append(
                    #         request_record.get_avg_completion_time_per_function(function_id)
                    #     )

                    # # Log resource utilization 
                    # resource_utils_record = info["resource_utils_record"]
                    # log_resource_utils(
                    #     logger_wrapper=logger_wrapper,
                    #     rm_name=rm, 
                    #     overwrite=False, 
                    #     episode=episode, 
                    #     resource_utils_record=resource_utils_record
                    # )

                    # # Log function throughput
                    # log_function_throughput(
                    #     logger_wrapper=logger_wrapper,
                    #     rm_name=rm, 
                    #     overwrite=False, 
                    #     episode=episode, 
                    #     function_throughput_list=function_throughput_list
                    # )
                    
                    episode_done = True
    
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

    # Log trends
    log_trends(
        logger_wrapper=logger_wrapper,
        rm_name=rm,
        overwrite=False,
        reward_trend=reward_trend,
        avg_completion_time_trend=avg_completion_time_trend,
        avg_completion_time_per_function_trend=avg_completion_time_per_function_trend,
        timeout_num_trend=timeout_num_trend,
        loss_trend=loss_trend,
    )

    
def lambda_rm_eval(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=10,
    hidden_dims_cpu=[64, 32],
    hidden_dims_memory=[64, 32],
    learning_rate=0.001,
    discount_factor=1,
    ppo_clip=0.2,
    ppo_steps=5,
    checkpoint_path="ckpt/best_model.pth",
    save_plot=False,
    show_plot=True,
):
    rm = "LambdaRM_eval"
    function_profile = profile.get_function_profile()

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    # Set up policy gradient agent
    pg_agent = PPO2Agent(
        observation_dim_cpu=env.observation_dim_cpu,
        observation_dim_memory=env.observation_dim_memory,
        action_dim_cpu=env.action_dim_cpu,
        action_dim_memory=env.action_dim_memory,
        hidden_dims_cpu=hidden_dims_cpu,
        hidden_dims_memory=hidden_dims_memory,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        ppo_clip=ppo_clip,
        ppo_steps=ppo_steps
    )

    # Restore checkpoint model
    pg_agent.load(checkpoint_path)
    
    # Trends recording
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    loss_trend = []
    avg_completion_time_per_function_trend = {}
    for function_id in function_profile.keys():
        avg_completion_time_per_function_trend[function_id] = []
    
    # Start random provision
    for episode in range(max_episode):
        env.reset()
        pg_agent.reset()

        actual_time = 0
        system_time = 0
        reward_sum = 0

        function_throughput_list = []
        
        episode_done = False
        while episode_done is False:
            actual_time = actual_time + 1

            timestep = timetable.get_timestep(system_time)
            if timestep is not None:
                for index, function_id in enumerate(timestep.keys()):
                    observation_cpu, observation_memory = env.get_observation(function_id)

                    if index == len(timestep) - 1:
                        time_proceed = True
                    else:
                        time_proceed = False

                    action_cpu, action_memory, value_pred_cpu, value_pred_memory, log_prob_cpu, log_prob_memory = pg_agent.choose_action(
                        observation_cpu=observation_cpu, 
                        observation_memory=observation_memory
                    )

                    reward, done, info = env.step(
                        time_proceed=time_proceed,
                        function_id=function_id,
                        action_cpu=action_cpu,
                        action_memory=action_memory
                    )

                    if system_time < info["system_time"]:
                        system_time = info["system_time"]
                        function_throughput_list.append(info["function_throughput"])

                    logger.debug("")
                    logger.debug("Actual timestep {}".format(actual_time))
                    logger.debug("System timestep {}".format(system_time))
                    logger.debug("Observation cpu: {}".format(observation_cpu))
                    logger.debug("Observation memory: {}".format(observation_memory))
                    logger.debug("Take action cpu: {}".format(action_cpu))
                    logger.debug("Take action memory: {}".format(action_memory))
                    logger.debug("Reward: {}".format(reward))
                
                    reward_sum = reward_sum + reward
            else:
                time_proceed = True
                function_id = None
                action_cpu = None
                action_memory = None

                reward, done, info = env.step(
                    time_proceed=time_proceed,
                    function_id=function_id,
                    action_cpu=action_cpu,
                    action_memory=action_memory
                )

                reward_sum = reward_sum + reward

                if done:
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
                    
                    reward_trend.append(reward_sum)
                    avg_completion_time_trend.append(avg_completion_time)
                    timeout_num_trend.append(timeout_num)

                    # Log average completion time per function
                    request_record = info["request_record"]
                    for function_id in avg_completion_time_per_function_trend.keys():
                        avg_completion_time_per_function_trend[function_id].append(
                            request_record.get_avg_completion_time_per_function(function_id)
                        )

                    # Log resource utilization 
                    resource_utils_record = info["resource_utils_record"]
                    log_resource_utils(
                        logger_wrapper=logger_wrapper,
                        rm_name=rm, 
                        overwrite=False, 
                        episode=episode, 
                        resource_utils_record=resource_utils_record
                    )

                    # Log function throughput
                    log_function_throughput(
                        logger_wrapper=logger_wrapper,
                        rm_name=rm, 
                        overwrite=False, 
                        episode=episode, 
                        function_throughput_list=function_throughput_list
                    )
                    
                    episode_done = True
    
    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(
            prefix_name=rm, 
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend, 
        )
    if show_plot is True:
        plotter.plot_show(
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend, 
            timeout_num_trend=timeout_num_trend, 
        )

    # Log trends
    log_trends(
        logger_wrapper=logger_wrapper,
        rm_name=rm,
        overwrite=False,
        reward_trend=reward_trend,
        avg_completion_time_trend=avg_completion_time_trend,
        avg_completion_time_per_function_trend=avg_completion_time_per_function_trend,
        timeout_num_trend=timeout_num_trend,
        loss_trend=None,
    )