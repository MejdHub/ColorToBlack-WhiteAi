# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:42:44 2021

@author: Am-pc
"""

import numpy as np

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
   return x * (1.0 - x) 

class NeuralNetwork:
    def __init__(self, w, x, y, z):
        self.w          = w
        self.x          = x
        self.y          = y
        
        self.input1     = w
        self.input2     = x        
        self.input3     = y
        
        # input layer to hidden layer 1 neuron 1
        
        self.weights1   = np.random.rand(self.input1.shape[1], 1)
        self.weights2   = np.random.rand(self.input2.shape[1], 1)
        self.weights3   = np.random.rand(self.input3.shape[1], 1)
        
        self.bias1      = 0
        
        
        # input layer to hidden layer 1 neuron 2
        
        
        self.weights4   = np.random.rand(self.input1.shape[1], 1)
        self.weights5   = np.random.rand(self.input2.shape[1], 1)
        self.weights6   = np.random.rand(self.input3.shape[1], 1)
        
        self.bias2      = 0
        
        
        # input layer to hidden layer 1 neuron 1
        
        
        self.weights7   = np.random.rand(self.input1.shape[1], 1)
        self.weights8   = np.random.rand(self.input2.shape[1], 1)
        self.weights9   = np.random.rand(self.input3.shape[1], 1)
        
        self.bias3      = 0
        
        """
        -------------------
        -------------------
        -------------------
        NEXT LAYER.ACTIVATE
        -------------------
        -------------------
        -------------------
        """
        
        
        # hidden layer 1 to hidden layer 2 neuron 1
        
        self.weights10  = np.random.rand(1,1)
        self.weights11  = np.random.rand(1,1)
        self.weights12  = np.random.rand(1,1)
        
        self.bias4      = 0
        
        # hidden layer 1 to hidden layer 2 neuron 2
        
        self.weights13  = np.random.rand(1,1)
        self.weights14  = np.random.rand(1,1)
        self.weights15  = np.random.rand(1,1)
        
        self.bias5      = 0
        
        
        """
        -------------------
        -------------------
        -------------------
        NEXT LAYER.ACTIVATE
        -------------------
        -------------------
        -------------------
        """
        
        
        # hidden layer 2 to output
        
        self.weights16   = np.random.rand(1,1)
        self.weights17   = np.random.rand(1,1)
        
        self.bias6       = 0
        
        
        """
        ---------------
        ---------------
        ---------------
        END OF AI SETUP
        ---------------
        ---------------
        ---------------
        """
        
        
        self.z          = z
        self.output     = np.zeros(self.z.shape)
        
        self.d          = 0
            
    def setwei (self, w1, w2, w3):
        self.weights1 = w1
        self.weights2 = w2
        self.weights3 = w3
                 

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input1, self.weights1) + 
                              np.dot(self.input2, self.weights2) +
                              np.dot(self.input3, self.weights3) + self.bias1) 
        
        self.layer2 = sigmoid(np.dot(self.input1, self.weights4) + 
                              np.dot(self.input2, self.weights5) +
                              np.dot(self.input3, self.weights6) + self.bias2) 
        
        self.layer3 = sigmoid(np.dot(self.input1, self.weights7) + 
                              np.dot(self.input2, self.weights8) +
                              np.dot(self.input3, self.weights9) + self.bias3)
        
        """END OF INPUT TO LAYER1"""
        
        self.layer4 = sigmoid(np.dot(self.layer1, self.weights10) +
                              np.dot(self.layer2, self.weights11) +
                              np.dot(self.layer3, self.weights12) + self.bias4)
        
        self.layer5 = sigmoid(np.dot(self.layer1, self.weights13) +
                              np.dot(self.layer2, self.weights14) +
                              np.dot(self.layer3, self.weights15) + self.bias5)
        
        """"" n.q.h """""
        
        #self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer4, self.weights16) +
                              np.dot(self.layer5, self.weights17) + self.bias6)

    def backprop(self, itrs):
        # application of the chain rule to find the derivation of the 
        # loss function with respect to weights
        
        d_weights1 = np.dot(self.input1.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights1.T) *                                                
                      sigmoid_derivative(self.layer1)))
        
        d_weights2 = np.dot(self.input2.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights2.T) *                                                
                      sigmoid_derivative(self.layer1)))
        
        d_weights3 = np.dot(self.input3.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights3.T) *                                                
                      sigmoid_derivative(self.layer1)))
        
        
        d_weights4 = np.dot(self.input1.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights4.T) *                                                
                      sigmoid_derivative(self.layer2)))
        
        d_weights5 = np.dot(self.input2.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights5.T) *                                                
                      sigmoid_derivative(self.layer2)))
        
        d_weights6 = np.dot(self.input3.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights6.T) *                                                
                      sigmoid_derivative(self.layer2)))
        
        
        d_weights7 = np.dot(self.input1.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights7.T) *                                                
                      sigmoid_derivative(self.layer3)))
        
        d_weights8 = np.dot(self.input2.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights8.T) *                                                
                      sigmoid_derivative(self.layer3)))
        
        d_weights9 = np.dot(self.input3.T, (2*np.dot((self.z - self.output) 
                    * sigmoid_derivative(self.output), self.weights9.T) *                                                
                      sigmoid_derivative(self.layer3)))
        
        
        
        
        """""""""""""""
        ---------------
        --END OF WOL1--
        ---------------
        """""""""""""""
        
        
        d_weights10 = np.dot(self.layer1.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights10.T) *
                     sigmoid_derivative(self.layer4)))
        
        d_weights11 = np.dot(self.layer2.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights11.T) *
                     sigmoid_derivative(self.layer4)))
        
        d_weights12 = np.dot(self.layer3.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights12.T) *
                     sigmoid_derivative(self.layer4)))
        
        d_weights13 = np.dot(self.layer1.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights13.T) *
                     sigmoid_derivative(self.layer5)))
        
        d_weights14 = np.dot(self.layer2.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights14.T) *
                     sigmoid_derivative(self.layer5)))
        
        d_weights15 = np.dot(self.layer3.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights15.T) *
                     sigmoid_derivative(self.layer5)))
        
        
        """""""""""""""
        ---------------
        --END OF WOL2--
        ---------------
        """""""""""""""
        
        d_weights16 = np.dot(self.layer4.T, (2*(self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output)))
        
        d_weights17 = np.dot(self.layer5.T, (2*(self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output)))
        
        
        
        """
        d_weights3 = np.dot(self.layer2.T, (2*(self.z - self.output) *                                                                          
                     sigmoid(self.output)))
        
        d_weights2 = np.dot(self.layer1.T, (2*np.dot((self.z - self.output) *                                                                          
                     sigmoid_derivative(self.output), self.weights3.T) *
                     sigmoid_derivative(self.layer2)))
        """
        
        
        
        self.weights1  += d_weights1
        self.weights2  += d_weights2
        self.weights3  += d_weights3
        self.weights4  += d_weights4
        self.weights5  += d_weights5
        self.weights6  += d_weights6
        self.weights7  += d_weights7
        self.weights8  += d_weights8
        self.weights9  += d_weights9
        self.weights10 += d_weights10
        self.weights11 += d_weights11
        self.weights12 += d_weights12
        self.weights13 += d_weights13
        self.weights14 += d_weights14
        self.weights15 += d_weights15
        self.weights16 += d_weights16
        self.weights17 += d_weights17
        
        self.feedforward()
        
        d_bias1 = 0#self.output - self.y/(itrs * 200)
        
        d_bias2 = 0#self.output - self.y/(itrs * 200)
        
        d_bias3 = 0#self.output - self.y/(itrs * 200)
        
        d_bias4 = 0#self.output - self.y/(itrs * 180)
        
        d_bias5 = 0#self.output - self.y/(itrs * 180)
        
        d_bias6 = 0#self.output - self.y/(itrs * 165)
        
        self.bias1 += d_bias1
        self.bias2 += d_bias2
        self.bias3 += d_bias3
        self.bias4 += d_bias4
        self.bias5 += d_bias5
        self.bias6 += d_bias6                
            
    def outputprod(self, a,b,c):
        self.input1 = a
        self.input2 = b
        self.input3 = c
        self.feedforward()
        return(self.output)
        #return(self.output - 0.2)
        self.input1 = self.w
        self.input2 = self.x
        self.input3 = self.y
        
    def train(self, iterations):
        for z in range(iterations):
            self.feedforward()
            self.backprop(iterations)
            if z+1 == iterations:
                n = self.outputprod(self.input1[0], self.input2[0], self.input3[0])
                self.prob = n/self.z[0]
        
    def giveAiDiagnosticsData(self):
        return(self.weights1) 
        return(self.weights2) 
        return(self.weights3)