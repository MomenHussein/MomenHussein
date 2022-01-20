# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import os #to make load and save actions
import torch
import torch.nn as nn #important tools of neural network
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Network(nn.Module):#to inherit tools from neural network
    
    def __init__(self,input_size,nb_action):
        super(Network,self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) #to make this full connection between the neurons and the input layer to the neurons of the hidden layer,full connections means : all the neurons of the input layer will all be connected to all the neurons of the hidden layer, so I used linear function to make it full connection, (in_features)the first argument in linear is the number of neurons of the first layer I want to connect them ,(out_features) the second argument is the number of the second layer I want to connect that is the layer at the right that is the hidden layer,(bias = true) third argument I wil keep it true to have a weight and one bias for each layer
        self.fc2 = nn.Linear(30,nb_action) #its full connection between the hidden layer and output layer,first parameter is the number of hidden layer , second parameter is the number of output layer
    
    def forward(self , state):
        x = F.relu(self.fc1(state)) #to activate the full connection with the hidden layer and save it in variable x
        q_values = self.fc2(x) #to nest the q_values to the out layers by using (x) the hidden layer
        return q_values
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
        
    def push(self , event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]   #to remove the oldest event to put the new event
            
    def sample(self , batch_size):
        samples = zip(*random.sample(self.memory , batch_size)) #using sample function from random class which I imported wich size of batch_size , zip(*) it is a reshape function , I used zip(*) because I want to reshape samples of the form first the state and second the action and third the reaward the 3 are in 1 parameter
        return map(lambda x: Variable(torch.cat(x,0)), samples) #it maps the samples to torch variables so I can return it and that will contain a sensor and gradient, first parameter is a function and the second parameter is a sequence , and the Variable which I caleed from torch will convert these samples to a torch variable an save them in x ,torch.cat(x,0) to concatenate the 3 samples(state,action,reward) in one parameter together in x and zero is the dimension(index) , and samples here it's which I want to apply lambda function on



class Dqn():
    
    def __init__(self,input_size,nb_action,gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size,nb_action) #to make it object from network class
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #because it has all tools to perform stochastic grid in the center, so it's contain some optimizers, the parameter of Adam is all the parameters that can customize me and optimizer and I put model in it just to connect the optimizer to our neural network model, lr is learning rate for the AI I make it small to give AI much time to explore and learn
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #it's a vector of 5 dimensions and they are 3 signals of the 3 sensors (right , left and straight) and orientation and minus orientation but for the torch I need more dimension I will put 1 more fake dimension for the batch and the last state will be the input of neural network , but the network can only accept batch of input, unsqueeze to input the fake dimension index(it will be in the first index)
        self.last_action = 0
        self.last_reward = 0
        
        def select_action(self,state):
            probs = F.softmax(self.model(Variable(state , volatile = True))*100) #T=100 , T is the tempreture ,soft max to calculate the max probability for the 3 actions of q_values to choose the best action the car desn't go to the sand, I put in it's parameter what I want to get the probability distribution the q_values are the output of the neural network ,  and the model have the q_values , volatile = true will include the gradients associated to this input states to the graph of all the conditionsof the end of that model so it will save for us some memory and improve the performance , tempreture parameter is the parameter that would allow us to modulate how the neural network will be sure of which action it should decide to play so it will be positive parameter and the closer it to zero the less sure the neural network will be when playing in action  and the higher the tempreture parameter is the more sure the neural network will be of the action it decides to play, T(tempreture) discribes the certainly of movement of AI to car , but when I increase the tempreture I also low the other probabilities (because AI depends on probability) this increasing the certainly
            action = probs.multinomial() #get random random draw from this distribution to get our final action, so I get the high chance to get the action that corresponds to the q_value that has the highest probability because that's exactly how the distribution works, probs = probability
            return action.data[0,0] #multinomial returns the pi torch cariable with a fake batch because it's fake dimension , I put actions.data to get the right values
          
        def learn(self , batch_state, batch_next_state, batch_reward, batch_action):
            outputs = self.model(batch_state).gather(1,batch_action).unsqueeze(1).squeeze(1) #we put 1 because we only wante the action that was chosen and then we add that action with this one in this action and we will gather each time the best action to play for each of the input states of the batch state, we put unsqueezed because we need the batch action have the same dimension of the state, and 1 instead of 0 because 0 correspond to the fake dimension of the state and 1 will correspond to the fake dimension of the actions , we need to kill this fake batch with a squeeze(last squeeze) because now we are out of the neural network we have our outputs but we don't want them in batch we want them in the symbol tensor in symbol vector(a vector of outputs) the batches just when we work in the neural network because the network is expecting the format of sensors into a batch
            next_outputs= self.model(batch_next_state).detach().max(1)[0] #we need the next state beacuse of target , the target = gamma*next output + max(what we want), next output is the result of our neural network when the batched next state is entering it as input, detach all the outputs of the model because we have several states in this batch next states that's the batch of all the next states in all the transitions taken from the random sample of our memory, since we are taking the max of these q values with respect to the action well we have to specify that it is with respect to the action and since the action is represented by the index 1, and the next state is represented by index 0 because the index 0 corresponds to the states therefore we need to add bracket[0]
            target = self.gamma*next_outputs + batch_reward #target = reward + gamma*next_outputs according to the actions 
            td_loss = F.smooth_l1_loss(outputs, target) #loss is the error of prediction, the function smoot improves the q_learning, and this parameters are predictions(outputs) and target
            self.optimizer.zero_grad() #we back propagators error back into the network to update the weights with stochastic gradient descent, zero.grad to re initialize the optimizer from one iteration to the other in the loop of this stochastic grad in the set, so we can perform backward propagation with our optimizer
            td_loss.backward(retain_variables=True) #to improve back propagation,using of variables=True is to free some memory, and we need to free some memory because we are going to go several times on the last so that will improve the training performance
            self.optimizer.step() #this will update the wights
            
        def update(self, reward, new_signal): #it's actually the last reward but I changed it's name from last_reward to reward because not to be confussed with the variable last reward that I made in the class
            new_state = torch.Tensor(new_signal).float().unsqueeze(0) #unsqueeze to make a fake dimension
            self.memory.push((self.last_state, new_state , torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #because last action contain 0 , 1 or 2
            action = self.select_action(new_state) #we play the new action after reaching the new state
            if len(self.memory.memory) > 100: #second memory is the memory of the class which identified in constructor which call replay memory function in it , the first memory is the memory which in replay memory function
                batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100) #to make learning better, 100 is the number of transitions
                self.learn(batch_state, batch_next_state, batch_reward, batch_action) #to make this learning happen , because if I have 100 transitions I will have 100 states , 100 new states , 100 actions and 100 rewards
            self.last_action = action #to update the value
            self.last_state = new_state
            self.last_reward = reward
            self.reward_window.append(reward)
            if len(self.reward_window) > 1000:
                del self.reward_window[0] #deleting the first element of reward window (I initialized reward window as array in constructor), I do it to be sure that the size of reward window won't be more than 1000 means of the last 100 rewards , because reward window calculate the mean of 100 reward, so it will be a window with fixed sized to make the training go well
            return action
    
    def score(self): #it computes the mean of all rewards
        return sum(self.reward_window)/(len(self.reward_window)+1.) #I put +1 because the length of rewards is deminator so I put +1 to make sure that it won't be equal to zero (because dividing by zero will give infinity and it will crash my system too)
    
    def save(self): #what I will save is the optimizer with the weights which I updated
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth') #I have here two keys one for the first object we want to save which is self.model and one second key for the second thing we want to save that is our self.optimizer , so I will use dictionary here to give them the names , last_brain.pth is the name of file which I save in
    
    def load(self):
        if os.path.isfile('last_brain.pth'): #I make sure that the file where I saved the last optimizer and model exist , os: operating system
            print("=> loading checkpoints...")
            checkpoint = torch.load('last_brain.pth') #it is to load the model and optimizer from the file
            self.model.load_state_dict(checkpoint['state_dict']) #to update all parameters of our model and all the weights , this method we inherited it from torch, and I pass to it the last point and the key of the model, and I need to do the same for the optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer']) # here I update the parameters of the optimizer
            print("done !")
        else:
            print("no checkpoint found...")
            