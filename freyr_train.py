import torch
import csv
from env import Environment
from params import WorkloadParameters, EnvParameters
from logger import Logger
from ppo2_agent import PPO2Agent
from params import *


#
# Policy gradient
#

def freyr_train(logger_wrapper):
    rm = "Freyr_train"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)

    # Set up policy gradient agent
    agent = PPO2Agent(
        state_dim=STATE_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        action_dim=ACTION_DIM,
        hidden_dims=HIDDEN_DIMS,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        ppo_clip=PPO_CLIP,
        ppo_epoch=PPO_EPOCH,
        value_loss_coef=VALUE_LOSS_COEF,
        entropy_coef=ENTROPY_COEF
    )

    # Record max sum rewards
    max_reward = -1e10
    max_reward_episode = 0

    # Trends recording
    reward_trend = []
    loss_trend = []
    csv_train = [["episode", "loss", "reward"]]
    csv_attn_weights = [
        [
            "episode", 
            "n_undone_request",
            "server_available_cpu",
            "server_available_memory",
            "function_avg_interval",
            "function_avg_cpu_peak",
            "function_avg_memory_peak",
            "function_avg_duration",
            "function_baseline",
            "cpu",
            "memory"
        ]
    ]

    # Start training
    exp_id = EXP_TRAIN[0]
    for episode in range(MAX_EPISODE_TRAIN):
        if exp_id != EXP_TRAIN[int(episode // (MAX_EPISODE_TRAIN/len(EXP_TRAIN)))]:
            # exp_id = EXP_TRAIN[int(episode // (MAX_EPISODE_TRAIN/len(EXP_TRAIN)))]

            # Retrain?
            if not USE_INCREMENTAL:
                agent = PPO2Agent(
                    state_dim=STATE_DIM,
                    action_dim=ACTION_DIM,
                    hidden_dims=HIDDEN_DIMS,
                    learning_rate=LEARNING_RATE,
                    discount_factor=DISCOUNT_FACTOR,
                    ppo_clip=PPO_CLIP,
                    ppo_epoch=PPO_EPOCH,
                    value_loss_coef=VALUE_LOSS_COEF,
                    entropy_coef=ENTROPY_COEF
                )

        # Set paramters for workloads
        workload_params = WorkloadParameters(
            azure_file_path=AZURE_FILE_PATH,
            exp_id=exp_id
        )

        # Set paramters for Environment
        env_params = EnvParameters(
            cluster_size=CLUSTER_SIZE,
            user_cpu_per_server=USER_CPU_PER_SERVER,
            user_memory_per_server=USER_MEMORY_PER_SERVER,
            keep_alive_window=KEEP_ALIVE_WINDOW,
            cpu_cap_per_function=CPU_CAP_PER_FUNCTION,
            memory_cap_per_function=MEMORY_CAP_PER_FUNCTION,
            memory_mb_limit=MEMORY_MB_LIMIT,
            interval=ENV_INTERVAL_LIMIT,
            fail_penalty=FAIL_PENALTY
        )

        # Set up environment
        env = Environment(
            workload_params=workload_params,
            env_params=env_params,
            incremental=int(episode // (MAX_EPISODE_TRAIN/len(EXP_TRAIN)))+1
            # incremental=0
        )

        # Record total number of events
        total_events = env.event_pq.get_total_size()

        # Reset logger, env, agent
        logger = logger_wrapper.get_logger(rm, False)
        state, mask, current_timestep, current_function_id = env.reset()
        agent.reset()

        actual_time = 0
        system_time = 0
        reward_sum = 0

        state_history = []
        mask_history = []
        action_history = []
        reward_history = []
        value_history = []
        log_prob_history = []

        attn_weights_history = []

        episode_done = False
        while episode_done is False:
            actual_time = actual_time + 1
            action, _, value_pred, log_prob, attn_weights = agent.choose_action(state, mask)
            next_state, next_mask, reward, done, info, next_timestep, next_function_id = env.step(
                current_timestep=current_timestep,
                current_function_id=current_function_id,
                action=env.decode_action(action)
            )

            if system_time < info["system_time"]:
                system_time = info["system_time"]

            # Record trajectories
            state_history.append(state.detach())
            mask_history.append(mask.detach())
            action_history.append(action.detach())
            reward_history.append(reward)
            value_history.append(value_pred.detach())
            log_prob_history.append(log_prob.detach())

            # Record attention weights
            attn_weights = attn_weights.detach()
            attn_weights_history.append(attn_weights)

            # logger.debug("")
            # logger.debug("Actual timestep {}".format(actual_time))
            # logger.debug("System timestep {}".format(system_time))
            # logger.debug("Take action: {}".format(action))
            # logger.debug("state: {}".format(state))
            # logger.debug("Reward: {}".format(reward))

            reward_sum = reward_sum + reward

            if done:
                if system_time < info["system_time"]:
                    system_time = info["system_time"]
                
                # Concatenate trajectories
                loss = 0
                state_history = torch.cat(state_history, dim=0)
                mask_history = torch.cat(mask_history, dim=0)
                action_history = torch.cat(action_history, dim=0)
                value_history = torch.cat(value_history).squeeze()
                log_prob_history = torch.cat(log_prob_history, dim=0)

                loss = loss + agent.update(
                    state_history=state_history,
                    mask_history=mask_history,
                    action_history=action_history,
                    reward_history=reward_history,
                    value_history=value_history,
                    log_prob_history=log_prob_history
                )

                # Save the min duration slo model
                if reward_sum > max_reward:
                    max_reward = reward_sum
                    max_reward_episode = episode + 1
                    agent.save(MODEL_SAVE_PATH + "max_reward.ckpt")

                logger.info("")
                logger.info("**********")
                logger.info("**********")
                logger.info("**********")
                logger.info("")
                logger.info("Running {}".format(rm))
                logger.info("Exp {}, Episode {} finished".format(exp_id, episode+1))
                logger.info("{} actual timesteps".format(actual_time))
                logger.info("{} system timesteps".format(system_time))
                logger.info("Total events: {}".format(total_events))
                logger.info("Total reward: {}".format(reward_sum))
                logger.info("Loss: {}".format(loss))
                logger.info("")
                logger.info("Max reward: {}, observed at episode {}".format(max_reward, max_reward_episode))
                
                reward_trend.append(reward_sum)
                loss_trend.append(loss)
                csv_train.append([episode+1, loss, reward_sum])

                # Avg attention weights
                # avg_weights = torch.mean(torch.cat(attn_weights_history, dim=0), 
                # print(attn_weights.squeeze(0).numpy().tolist())
                csv_attn_weights.append([episode+1] + attn_weights.squeeze(0).numpy().tolist())
                # print(csv_attn_weights)
                
                episode_done = True
            
            state = next_state
            mask = next_mask
            current_timestep = next_timestep
            current_function_id = next_function_id
    
    with open("logs/{}_{}_{}_no_attn_weights.csv".format(rm, exp_id, MAX_EPISODE_TRAIN), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_attn_weights)

    with open("logs/{}_{}_{}_no_attn.csv".format(rm, exp_id, MAX_EPISODE_TRAIN), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_train)


if __name__ == "__main__":
    # Set up logger wrapper
    logger_wrapper = Logger()

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    print("Start training...")

    freyr_train(logger_wrapper=logger_wrapper)

    print("")
    print("Training finished!")
    print("")
    print("**********")
    print("**********")
    print("**********")
