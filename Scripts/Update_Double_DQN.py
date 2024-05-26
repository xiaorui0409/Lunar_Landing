from collections import namedtuple
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import time
import gymnasium as gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import logging
import scanpy


class Q_Network(nn.Module):
    def __init__(self, state_size, num_actions, seed):
        super(Q_Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))       
        x = self.relu(self.layer2(x))
        x = self.output(x)

        return x
    
class Target_Q_Network(nn.Module):
    def __init__(self, state_size, num_actions,):
        super(Target_Q_Network, self).__init__()

        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))       
        x = self.relu(self.layer2(x))
        x = self.output(x)

        return x

def transform_tensor(data, dtype, device):
    return torch.tensor(data, dtype=dtype, device=device)


class ReplayBuffer:
    def __init__(self, capacity,device):
        self.buffer = deque(maxlen = capacity) 
        self.device=device
    
    #### Store the experience
    def push(self, state, action, reward, next_state, terminated, truncated, info):
        experience = (state, action, reward, next_state, terminated, truncated, info)
        self.buffer.append(experience)

    def sample(self, batch_size):

        if len(self.buffer) < batch_size:
            raise ValueError("Not enough elements in the buffer to sample!!!!   ")
        

        #### batch sampel from buffer
        sample_experience = random.sample(self.buffer, batch_size)
        ### unpacking operator
        state, action, reward, next_state, terminated, truncated, info = zip(*sample_experience)

        # Convert to numpy arrays and then to tensors
        state = np.array([np.array(s) for s in state], dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)
        next_state = np.array([np.array(s) for s in next_state], dtype=np.float32)
        terminated = np.array([d for d in terminated], dtype=bool)
        truncated = np.array([t for t in truncated], dtype=bool)

        return (torch.tensor(state, dtype=torch.float32, device=self.device),
            torch.tensor(action, dtype=torch.long, device=self.device),
            torch.tensor(reward, dtype=torch.float32, device=self.device),
            torch.tensor(next_state, dtype=torch.float32, device=self.device),
            torch.tensor(terminated, dtype=torch.bool, device=self.device),
            torch.tensor(truncated, dtype=torch.bool, device=self.device),
            info)


    def __len__(self):
        return len(self.buffer)

def compute_loss(sampled_state, sampled_action, sampled_reward, sampled_next_state, terminated_sample, sampled_truncated, gamma, q_network, target_q_network,device):
   
    q_network.eval()
    target_q_network.eval()

    with torch.no_grad():
        ###### Double Deep Q Networks
        ### use "q_network" to select the action for next_state
        action_select = q_network(sampled_next_state).max(1)[1] ## return max action index
        ### Use  "target_q_network" to calculate the return value based on the selected action index from previous step
        qsa_next = target_q_network(sampled_next_state).gather(1, action_select.unsqueeze(1)).squeeze(1)
    ### calcaute the "y_targets"
    y_targets = sampled_reward + gamma*(qsa_next)*(1 - terminated_sample.int())

    ### compute the predicted value from "q_network" for the current_state and action
    q_network.train()
    predicted_values = q_network(sampled_state).to(device).gather(1, sampled_action.unsqueeze(1)).squeeze(1) ##final result(bs, )

    ##### calcualte the loss
    mse_loss = torch.nn.functional.mse_loss(y_targets, predicted_values)

    return mse_loss


def soft_update(target_q_network, q_network, tau):
    ### zip the parameters of target_q_network  & q_network
    for target_param, source_param in zip(target_q_network.parameters(), q_network.parameters()):
        ### gradually update the target-q_network
        target_param.data.copy_(
            target_param.data*(1-tau) + source_param.data*tau  ##
        )

def hard_update(target_model, model):
    target_model.load_state_dict(model.state_dict())


def update_model(sampled_state, sampled_action, sampled_reward, sampled_next_state, terminated_sample, sampled_truncated, gamma, q_network, target_q_network,device,optimizer, tau, t):
    loss = compute_loss(sampled_state, sampled_action, sampled_reward, sampled_next_state, terminated_sample, sampled_truncated, gamma, q_network, target_q_network,device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### implement the "soft_updateing"    
    # if t % 800 == 0 and t > 0:
    #     soft_update(target_q_network, q_network, tau)

    ###### implement the "hard_update"    
    if t % 200 == 0 and t > 0:
        hard_update(target_q_network, q_network)


def choose_action(state, epsilon, q_network, num_actions, device):
    if random.random() < epsilon: ## explore
        return random.randint(0, num_actions - 1)
    else: ## exploit
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        state_tensor = state_tensor.unsqueeze(0)
        return q_network(state_tensor).max(1)[1].item()


def plot_progress(total_point_history, average_last100_points_list):
    plt.figure(figsize=(12, 6))  # Set the figure size for better visibility
    
    # Plot total points per episode
    plt.plot(total_point_history, label='Total Rewards per Episode', color='blue', alpha=1.0)  

    # Plot the average of the last 100 episodes
    if len(average_last100_points_list) > 0:
        plt.plot(average_last100_points_list, label='Average of Last 100 Episodes', color='red', linewidth=2)

    plt.xlabel('Episodes',fontsize=14, fontweight='bold')  
    plt.ylabel('Total Rewards',fontsize=14, fontweight='bold')  
    plt.title('Agent Training Progress', fontsize=16, fontweight='bold')  
    plt.legend()  # Add a legend to explain what the plot lines represent
    plt.grid(True)  # Turn on the grid for better readability of the plot
    plt.show()  


def train_agent(env,capacity, num_episodes, max_timesteps_per_episode, epsilon, epsilon_decay, epsilon_min, batch_size, gamma, 
                q_network, target_q_network, device, optimizer, tau,num_p_av,num_actions):
    memory_buffer = ReplayBuffer(capacity=capacity, device=device)  # Example capacity
    total_point_history = []
    average_last100_points_list = []

    for episode in range(num_episodes):
    ### initilize the environment state, reset the environment
        state, _ = env.reset()  ## extract the STATE from the tuple
        total_rewards = 0

        # if episode %500 == 0: # Render the environment at each step
        #         env.render()  

        for t in range(max_timesteps_per_episode):  ###
            #### decesion to "explore" or "exploitation"
            action = choose_action(state, epsilon, q_network, num_actions, device)

            next_state, reward, terminated, truncated, info = env.step(action) 

            ### store these result in "BUffer"    state, action, reward, next_state, done, truncated, info
            memory_buffer.push(state, action, reward, next_state, terminated, truncated, info) 
    
            ### update the state
            state = next_state.copy()
            total_rewards += reward

            ### train the model
            if len(memory_buffer) >= batch_size:
                ### random sample from buffer to break the correlation from the consecutive experience
                sampled_state, sampled_action, sampled_reward, sampled_next_state, terminated_sample, sampled_truncated, sampled_info = memory_buffer.sample(batch_size)  ####### 
                update_model(sampled_state, sampled_action, sampled_reward, sampled_next_state, terminated_sample, sampled_truncated, gamma, q_network, target_q_network,device,optimizer, tau, t,)
               
            if terminated or truncated:
                break
    
        total_point_history.append(total_rewards)
        ### New modified code:
        average_last100_points = np.mean(total_point_history[-num_p_av:])
        average_last100_points_list.append(average_last100_points)

        if (episode) %100 == 0  and episode > 0:
            print(f"Episode {episode}:  Average of the last 100 episodes: {average_last100_points:.2f}")

        if average_last100_points >=200: 
            print(f"\n\nEnvironment solved in {episode} episodes!\n")
            torch.save(q_network.state_dict(), 'lunar_lander_model_200.path')
            break

        #### Update the Epsilon greedy policy (Epsilon gradually decreased)  ### minmium:0.01
        epsilon = max(epsilon_min, epsilon_decay*epsilon)

    #env.close()
    
    return total_point_history, average_last100_points_list



def choose_action_inference(state, trained_q_network, device):

    state_tensor = torch.tensor(state, dtype=torch.float).to(device)
    state_tensor = state_tensor.unsqueeze(0)
    
    trained_q_network.eval()
    with torch.no_grad():
        action = trained_q_network(state_tensor).max(1)[1].item()
            
    return action


def make_video(env_name, trained_q_network, device, output_directory='videos_new_100', num_episodes=151,record_freq=30):
    env = gym.make(env_name,render_mode='rgb_array')
    env = RecordVideo(env, video_folder=output_directory,   episode_trigger=lambda episode: episode % record_freq == 0)

    ### "video_length" how long the video records in terms of timesteps per episode
    for _ in range(num_episodes):

        state, _ = env.reset()
        truncated =False
        terminated = False

        while not (terminated or truncated):
            action = choose_action_inference(state, trained_q_network, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state.copy()

    env.close()





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Set environment and hyperparameters
    env_name = "LunarLander-v2"
    env = gym.make(env_name, )   ## render_mode='human'


    state_size = 8
    num_actions = 4
    capacity = 200000  
    batch_size = 128  
    num_episodes = 1000  
    max_timesteps_per_episode = 1100 #1000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99    
    tau = 0.005 #0.002
    num_p_av = 100


    #### Initialize the Q_Network
    q_network = Q_Network(state_size, num_actions,seed=42).to(device)
    ####Initialization of Target Network:
    target_q_network = Target_Q_Network(state_size, num_actions).to(device)
    #### Target Network share the same weights as q_network
    target_q_network.load_state_dict(q_network.state_dict())
    optimizer = optim.AdamW(q_network.parameters(), lr=1e-3)

    ### Compute the training time
    start = time.time()
    ### Train the agent
    total_point_history, average_last100_points_list = train_agent(env,capacity, num_episodes, max_timesteps_per_episode, epsilon, epsilon_decay, epsilon_min, batch_size, gamma, 
                                                                   q_network, target_q_network, device, optimizer, tau,num_p_av,num_actions)

    ### Plot history to evaluate how the agent improved during training
    plot_progress(total_point_history, average_last100_points_list)

    total_time = time.time() - start
    print(f"TOTAL RUN TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")




    ####### create animation to visualize the agent interacting with the environment using the trained -Q_Network
    print("Starting create a vidoe________\n")
    trained_q_network = Q_Network(state_size, num_actions,seed=42).to(device)
    trained_q_network.load_state_dict(torch.load('lunar_lander_model_200.path'))

    make_video(env_name, trained_q_network, device, output_directory='videos_new_100', num_episodes=151,record_freq=30)



if __name__=="__main__":
    main()







