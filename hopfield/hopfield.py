# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:25:46 2020

@author: gn-00
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


###############################################
class hopfield:
    def __init__(self, input_shape):
        self.train_data = []
        self.W = np.zeros([input_shape[0]*input_shape[1],input_shape[0]*input_shape[1]],dtype=np.int8)
        
    def addTrain(self,img_dir):
        img = plt.imread(img_dir)
        img = np.mean(img,axis=2)
        if img.shape != (32,32):
            print('Error: image shape ',img.shape,' is not (32,32)')
            return
        img_mean = np.mean(img)
        img = np.where(img < img_mean,-1,1)
        train_data = img.flatten()
        for i in trange(train_data.size):
            for j in range(i,train_data.size):
                if i==j:
                    self.W[i][j] = 0
                else:
                    w_ij = train_data[i]*train_data[j]
                    self.W[i][j] += w_ij
                    self.W[j][i] += w_ij
    
    #                
    def update(self,state,idx=None):
        if idx==None:
            # state = np.matmul(self.W,state)
            # state = np.where(state<0,-1,1)
            new_state = np.matmul(self.W,state)
            new_state[new_state < 0] = -1
            new_state[new_state > 0] = 1
            new_state[new_state == 0] = state[new_state == 0]
            state = new_state
        else:
            # state[idx] = np.matmul(self.W[idx],state)
            # state[idx] = np.where(state[idx] < 0,-1,1)
            new_state = np.matmul(self.W[idx],state)
            if new_state < 0:
                state[idx] = -1
            elif new_state > 0:
                state[idx] = 1
        return state
    
    #
    def predict(self,mat_input,iteration,asyn=False,async_iteration=200):
        input_shape = mat_input.shape
        fig,axs = plt.subplots(1,1)
        axs.axis('off')
        print(input_shape)
        graph = axs.imshow(mat_input*255,cmap='binary')
        mat_input = np.where(mat_input < 0.5,-1,1)
        fig.canvas.draw_idle()
        plt.pause(1)
        e_list = []
        
        e = self.energy(mat_input.flatten())
        e_list.append(e)
        state = mat_input.flatten()
        
        if asyn:
            print('Starting asynchronous update with ',iteration,' iterations')
            for i in range(iteration):
                for j in range(async_iteration):
                    idx = np.random.randint(state.size)
                    state = self.update(state,idx)
                state_show = np.where(state < 1,0,1).reshape(input_shape)
                graph.set_data(state_show*255)
                axs.set_title('Async update Iteration #%i' %i)
                fig.canvas.draw_idle()
                plt.pause(0.25)
                new_e = -0.5*np.matmul(np.matmul(state.T,self.W),state)
                print('Iteration#',i,', Energy: ',new_e)
                if new_e == e:
                    print('Energy remain unchanged, update will now stop.')
                    break
                e = new_e
                e_list.append(e)
        else:
            print('Starting synchronous update with ',iteration,' iterations')
            for i in range(iteration):
                state = self.update(state)
                state_show = np.where(state < 1,0,1).reshape(input_shape)
                graph.set_data(state_show*255)
                axs.set_title('Sync update Iteration #%i' %i)
                fig.canvas.draw_idle()
                plt.pause(0.5)
                new_e = -0.5*np.matmul(np.matmul(state.T,self.W),state)
                print('Iteration#',i,', Energy: ',new_e)
                if new_e == e:
                    print('Energy remain unchanged, update will now stop.')
                    break
                e = new_e
                e_list.append(e)
        print('Iteration completed, update will now stop.')
        plt.pause(1)
        plt.close()
        return np.where(state < 1,0,1).reshape(input_shape),e_list
    
    def energy(self,o):
        e = -0.5*np.matmul(np.matmul(o.T,self.W),o)
        return e
