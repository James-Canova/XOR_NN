#main.py  (version 0)

#Micropython and Raspberry Pi Pico 
#Date created: 19 January 2024
#last updated: 19 February 2025

#James Canova
#jscanova@gmail.com

#Based on:
#Rashid, Make your own Neural Network
# https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/

#This program solves two input (one output) XOR logic using a neural network

import micropython
from machine import Pin
from ulab import numpy as np  #for matrix operations
import random as rnd          #to initialize weights and biases

#--------------------------------------------------------
#Hyperparameters (i.e. they control the solution)
#note: these were selected by trial and error
LEARNING_RATE = 0.08
EPOCHS = 20000

#initialize random number generator
rnd.seed(30)


#setup LEDs
#red LED on: not trained
#green LED on: trained
#blue LED on: output == 1
#yellow LED on: output == 0
ledRed = Pin(16, Pin.OUT)
ledGreen = Pin(18, Pin.OUT)
ledBlue = Pin(26, Pin.OUT)
ledYellow = Pin(28, Pin.OUT)


#setup slider switches for inputs
slider1 = Pin(14, Pin.IN, Pin.PULL_UP)
slider2 = Pin(15, Pin.IN, Pin.PULL_UP)


#activation function
def sigmoid (x):
    
    return 1/(1 + np.exp(-x))

pass


#for initializing weights and biases between and including:-1, 1 
#shape contains dimensions of required matrix
def create_random_array(shape):

    new_array = np.zeros(shape)

    nRows = shape[0]
    nColumns = shape[1]

    for i in range(nRows):
        for j in range(nColumns):
            new_array[i][j] += rnd.uniform(-1, 1)
    return new_array


#Neural network class
class neuralNetwork:
    # initialise the neural network
    # note: one hidden layer only
    # inputnodes = number of input nodes 
    # hiddennodes = number of hidden nodes
    # ouptutnodes = numper of output nodes
    
    #member functions:
    #-init
    #-train
    #-infer
    
    #-----------------------------------------------------------------------------
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, epochs): 

        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.epochs = epochs

        #Initialize weights and biases with random numbers, -1 <= num <= 1 
        self.wih = create_random_array((hiddennodes, inputnodes))   # 2 x 2
        self.who = create_random_array((outputnodes, hiddennodes))  # 1 x 2

        self.bih = create_random_array((hiddennodes,1))  # 2 x 1
        self.bho = create_random_array((outputnodes,1))  # 1 x 1

       # learning rate
        self.lr = learningrate

        # number of epochs
        self.epochs = epochs
        
        #flag for training
        self.bTrained = False
        

    #---------------------------------------------------------------------------
    # train
    # inputs is a 4 x 2 numpy array, each row is a pair of input values
    # targets is a 4 x 1 numpy array
    def train(self, inputs, targets):

        # interate through epochs 
        for c1 in range(self.epochs):

          epoch_cost = 0.0  # initialize cost per epoch (for plotting)  

          # interate through 4 inputs
          for c2 in range(inputs.shape[0]):  #inputs.shape[0] equals the number of input pairs which is 4

            input = inputs[c2,:]   # 1 x 2 
            input = input.reshape((2, 1)) # 2 x 1

            target = targets[c2,:]   #target is a 1D numpy array

            #forward propagation-----
            #calculate hidden outputs from inputs
            hidden_sums = np.dot(self.wih, input) + self.bih # 2 x 2 . 2 x 1 + 2 x 1 = 2 x 1 
            hidden_outputs = sigmoid(hidden_sums)  # 2 x 1

            #calculate predicted output from hidden outputs
            output_sum = np.dot(self.who, hidden_outputs) + self.bho # 1 x 2 . 2 x 1 + 1 x 1 = 1 x 1 
            final_output = sigmoid(output_sum)  # 1 x 1
     
        
            #backward propagation-----
            #update weights for hidden to output layer
            output_error = target - final_output   #1 x 1 - 1 x 1 
            dWho = self.lr * np.dot((output_error * final_output * (1.0 - final_output)), hidden_outputs.T) # 1 x 1 . 1 x 2 = 1 x 2
            self.who += dWho # 1 x 2

            #update bias for output layer
            dbho = self.lr * output_error * (final_output * (1.0 - final_output)) # 1 x 1
            self.bho += dbho # 1 x 1

            #update weights for hidden layer
            hidden_error = np.dot(self.who.T, output_error) # 2 x 1 . 1 x 1 = 2 x 1 [Rashid, pp.81, 82, 140]
            dWih = self.lr * np.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)), input.T) # 2 x 2 
            self.wih += dWih  # 2 x 2  

            #update biases for input to hidden layer
            dbih = self.lr * hidden_error * (hidden_outputs * (1.0 - hidden_outputs)) # 2 x 1
            self.bih += dbih  # 2 x 1

        self.bTrained = True

    #---------------------------------------------------------------------------
    # to infer output from inputs
    def infer(self, input):

      #calculate hidden outputs from inputs
      hidden_sums = np.dot(self.wih, input) + self.bih  # 2 x 2 . 2 x 1 + 2 x 1 = 2 x 1  
      hidden_outputs = sigmoid(hidden_sums)  # 2 x 1

      #calculate predicted output from hidden outputs
      output_sum = np.dot(self.who, hidden_outputs) + self.bho   # 1 x 2 . 2 x 1 + 1 x 1 = 1 x 1 
      inferred_output = sigmoid(output_sum)  # 1 x 1

      return inferred_output # 1 x 1


#====================================================================
#main program
#set LEDs to indicate neural network has not been trained
ledRed.on()
ledGreen.off() 
ledBlue.off() 
ledYellow.off()    
    

# initialize neural network------
# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 2
output_nodes = 1


# create instance of neural network
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, LEARNING_RATE, EPOCHS)


#train the neural network-------
#define the training dataset, which are inputs and targets (outputs)
#define inputs and targets for training and infer
inputs_array= np.array([[0,0],[0,1],[1,0],[1,1]])  # 4 x 2
targets_array = np.array([[0],[1],[1],[0]])	# 4 x 1

#train
nn.train(inputs_array, targets_array);

#indicate that the neural network has been trained
ledRed.off()
ledGreen.on()  


#main loop==================================================================
while True:
    
    #read state of slider switches (inputs)
    nValueInput0 = slider1.value()
    nValueInput1 = slider2.value()      
    
    #format inputs for passing to neural network function for inferring results
    inputs_list = np.zeros((2,1))
    inputs_list[0,0]= nValueInput0;
    inputs_list[1,0]= nValueInput1;       
    
    #infer result (0 or 1)
    final_output= nn.infer(inputs_list)
    nInferredResult = int(round(final_output[0,0])) #round to nearest of 0 or 1
    
    #display result using LEDS
    if nInferredResult == 1: 
        ledBlue.on()
        ledYellow.off()

    elif nInferredResult == 0:  
        ledYellow.on()
        ledBlue.off()          

    else:
        pass

    pass

pass    #end of infinite loop

