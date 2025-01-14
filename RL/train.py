
import os
import time
import torch
import random
import argparse
import numpy as np
import networkx as nx
from distutils.util import strtobool
from datetime import datetime

from torch_geometric.data import Batch
# from torch.utils.tensorboard import SummaryWriter
from env.env_zx import QuantumCircuitSimplificationEnv as EnvZx
from RL.PPO import PPO
from RL.replaybuffer import ReplayBuffer
from RL.test_env import TestEnv
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--env_name", type=str, default="zx-calculus", help=" Name of Env")
    parser.add_argument('--max_episode_step', type=int, default=400, help='size of circuit')
    parser.add_argument("--max_train_steps", type=int, default=int(400), help=" Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=5, help="Minibatch size")
    parser.add_argument("--c_in_p", type=int, default=14, help="Dim of network")  
    parser.add_argument("--c_in_critic", type=int, default=14, help="Dim of network")  
    parser.add_argument("--edge_dim", type=int, default=3, help="Dim of network")  
    parser.add_argument("--edge_dim_critic", type=int, default=3, help="Dim of network")  

    parser.add_argument("--hidden_width", type=int, default=32,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.02, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=2, help="PPO parameter")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="policy entropy")
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='cuda or cpu')
    parser.add_argument('--obs_shape', type=int, default=3000, help='size of circuit')
    args = parser.parse_args()
    args.device = torch.device(args.device)

    return args


def train():
    print('==* Initialization *==')
    # Arguments for running the program
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic  # Set PyTorch cuDNN backend to use deterministic algorithms for reproducibility

    # Checkpoint file folder
    model_dir = 'data/log/model_checkpoints'  # Save the model in the model_checkpoints folder at the same level as src
    os.makedirs(model_dir, exist_ok=True)  # Ensure the log_dir directory exists

    # Initialize the environment
    chip_graph = nx.Graph()
    chip_graph.add_nodes_from([0, 1, 2])  # Assume there are 3 qubits
    env = EnvZx(chip_graph)  # Use the EnvZx environment
    env_evaluate = EnvZx(chip_graph)  # Use the same environment for evaluation

    # Initialize the agent
    agent = PPO(args)

    # Initialize the Replay Buffer
    replay_buffer = ReplayBuffer(args)

    for input_file in env.all_files:
        file_path = os.path.join(env.input_folder, input_file)
        s, _ = env.reset(file_path)  # Reset the environment and load the circuit file

        # For storing data during training
        reward_list = []
        actor_loss_list = []
        critic_loss_list = []

        # Training process
        print('==** Training... **==')
        start_time = time.time()
        total_steps, episode_num = 0, 0
        while total_steps < args.max_train_steps:  # For each episode
            episode_num += 1
            episode_steps = 0
            done = False
            s, _ = env.reset(file_path)

            while not done and episode_steps < args.max_episode_step:
                episode_steps += 1
                a, a_logprob, _, identifier = agent.choose_action(Batch.from_data_list([s]))
                s_, r, done, _, _ = env.step(identifier)

                if done and episode_steps != args.max_episode_step:
                    dw = True
                else:
                    dw = False

                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # Update
                if replay_buffer.count >= args.batch_size:
                    print("")

                    # Update
                    actor_loss, critic_loss = agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0

                    # Evaluate the policy
                    evaluate_reward = evaluate_policy(args, env_evaluate, agent)
                    # Append the data to the respective lists
                    print(f"Evaluate Reward: {evaluate_reward}")
                    print(f"Critic Loss: {critic_loss}")

                    reward_list.append(evaluate_reward)
                    actor_loss_list.append(np.mean(actor_loss))
                    critic_loss_list.append(np.mean(critic_loss))
        # After training, extract and save the simplified quantum circuit
    try:
        simplified_gate_sequence = env.extract_gates_directly(env.zx_graph)
        output_file_path = os.path.join(env.output_folder, f'simplified_{input_file}')
        with open(output_file_path, 'w') as file:
            file.write(simplified_gate_sequence)
        print("[INFO] Simplified QASM instruction set has been saved to:", output_file_path)
    except Exception as e:
        print("[ERROR] Error saving the simplified quantum circuit as QASM:", e)            

    end_time = time.time()
    elapsed_time = end_time - start_time
    # Convert elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print('==* Training End *==')
    print(f"Expenditure: {hours}h: {minutes}min: {seconds}s")

    # Plot the reward curve
    plt.figure()
    plt.plot(reward_list, label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.grid()
    plt.savefig('reward_curve.png')  # Save the plot to a file
    plt.show()  # Display the plot

    # Plot the actor's loss curve
    plt.figure()
    plt.plot(actor_loss_list, label='Actor Loss')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Actor Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig('actor_loss_curve.png')  # Save the plot to a file
    plt.show()  # Display the plot

    # Plot the critic's loss curve
    plt.figure()
    plt.plot(critic_loss_list, label='Critic Loss')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Critic Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig('critic_loss_curve.png')  # Save the plot to a file
    plt.show()  # Display the plot


def evaluate_policy(args, env, agent, times=30):
    """ test the efficiency of the current model """
    evaluate_reward = 0
    for i in range(times):
        done = False
        s, _ = env.reset()
        episode_steps = 0
        while not done and episode_steps < args.max_episode_step:
            episode_steps += 1
            _, _, _, identifier = agent.choose_action(Batch.from_data_list([s]))
            s_, r, done, _, _ = env.step(identifier)
            s = s_
            evaluate_reward += r
        print("evaluate_reward", evaluate_reward)
    print('')

    return evaluate_reward / times

