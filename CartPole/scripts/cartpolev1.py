import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time

import os.path
import numpy

import matplotlib.pyplot as plt
from atari_wrappers import make_atari, wrap_deepmind

plt.style.use('ggplot')

videosDir = './RLvideos/'


# env_id = "PongNoFrameskip-v4"
# env = make_atari(env_id)
# env = wrap_deepmind(env)

env = gym.make('CartPole-v0' )

env = gym.wrappers.Monitor(env, videosDir , force=True , video_callable=lambda episode_id: episode_id%200==0)






seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)




###### PARAMS ######
learning_rate = 0.01
num_episodes = 800
gamma = 0.90



egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 1500

report_interval = 50
score_to_solve = 190



file2save = 'pong_save.pth'
save_model_frequency = 50000
resume_previous_training = False



device = "cuda" if torch.cuda.is_available() else "cpu"

####################





def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

def load_model():
    return torch.load(file2save)

def save_model(model , path = file2save):
    torch.save(model.state_dict(), path)








def plot_results():
    plt.figure(figsize=(12,5))
    plt.title("Rewards")
    plt.plot(rewards_total, alpha=0.6, color='red')
    plt.savefig("Pong-results.png")
    plt.close()







class Q_Network(nn.Module):
    def __init__(self , number_of_inputs , number_of_outputs):
        super(Q_Network, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,64)
        self.linear2 = nn.Linear(64,number_of_outputs)

        self.activation = nn.Tanh()
        #self.activation = nn.ReLU()
        
        
    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        return output2





class QNet_Agent():
    def __init__(self , number_of_inputs , number_of_outputs):

        self.nn = Q_Network(number_of_inputs , number_of_outputs).to(device)


        self.loss_func = nn.MSELoss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        # self.Scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer , factor= 0.2 , patience=4000 , verbose=True)
        
        self.number_of_frames = 0
        
        if resume_previous_training and os.path.exists(file2save):
            print("Loading previously saved model ... ")
            self.nn.load_state_dict(load_model())
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                
                state = torch.tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()        
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self , state, action, new_state, reward, done):
        

 
        state = torch.tensor(state).to(device)
        new_state = torch.tensor(new_state).to(device)
        reward = torch.tensor([reward]).to(device)



        if done:
            target_value = reward

        else:
            new_state_values = self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values)
            target_value = reward + gamma * max_new_state_values


        

  
        predicted_value = self.nn(state)[action]

        loss = self.loss_func(predicted_value, target_value)
       

    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        
        if self.number_of_frames % save_model_frequency == 0:
            print("** save the model **")
            save_model(self.nn)
        
        self.number_of_frames += 1



number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n


qnet_agent = QNet_Agent(number_of_inputs , number_of_outputs)

steps_total = []

frames_total = 0 
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    
    while True:
        
        step += 1
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        action = qnet_agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)

        qnet_agent.optimize(state, action, new_state, reward, done)
        
        state = new_state
        
        if done:
            steps_total.append(step)
            
            mean_reward_100 = sum(steps_total[-100:])/100
            if (mean_reward_100 > score_to_solve and solved == False):

                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
                print("save the after solved the game")
                save_model(qnet_agent.nn , path="solved_cartpole.pt")

            if (i_episode % report_interval == 0):
                
                
                
                print("\n*** Episode %i *** \
                      \nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                      \nepsilon: %.2f, frames_total: %i" 
                  % 
                  ( i_episode,
                    report_interval,
                    sum(steps_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(steps_total)/len(steps_total),
                    epsilon,
                    frames_total
                          ) 
                  )
                  
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



            break
        

print("\n\n\n\nAverage reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))
if solved:
    print("Solved after %i episodes" % solved_after)


plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

env.close()
env.env.close()















