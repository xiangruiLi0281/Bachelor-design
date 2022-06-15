#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy
import scipy.special
import matplotlib.pyplot as plt
from tkinter import _flatten
from itertools import chain
import imageio
import scipy.misc
from scipy.ndimage import uniform_filter
import cv2


# In[30]:


#Block of initialization
class neuralNetwork :
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
# set number of nodes in each input, hidden,output layer
         self.iNodes=inputNodes
         self.hNodes=hiddenNodes
         self.oNodes=outputNodes
 #learning rate
         self.lr=learningRate
#set up weight matrix which meet the require of Gaussian distribution
#function(argument1,argument2,argument3)
#argument1 refers to means of Gaussian distribution 
#argument2 refers to standar deviation
#argument3 refers to dim of matrix 
#weight input to hidden layer
         self.Wih=numpy.random.normal(0.0,pow(self.hNodes,-0.5),(self.hNodes,self.iNodes))
         self.b1=numpy.random.normal(0.0,pow(self.hNodes,-0.5),(self.hNodes,1))
#weight hidden to output layer
         self.Who=numpy.random.normal(0.0,pow(self.oNodes,-0.5),(self.oNodes,self.hNodes))
         self.b2=numpy.random.normal(0.0,pow(self.oNodes,-0.5),(self.oNodes,1))
#Activation function sigmoid()
         self.activation_function_softmax= lambda X: numpy.exp(X-max(X))/numpy.sum(numpy.exp(X-max(X)))
         self.activation_function_sigmoid=lambda X: scipy.special.expit(X)
#Block of training
    def train(self, input_list, targets_list):
        #transform list to array
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        #input of layer hidden refers to O=Wih·I
        hidden_inputs = numpy.dot(self.Wih,inputs)+self.b1
        #via activation function 
        hidden_outputs=self.activation_function_sigmoid(hidden_inputs)
        #input of layer output refers to O=Wh0·hiddenl_output
        final_inputs = numpy.dot(self.Who,hidden_outputs)+self.b2
        #via activation function 
        final_outputs=self.activation_function_softmax(final_inputs)
        #define whole error
        output_errors=-targets*numpy.log(final_outputs)
        #print(sum(output_errors))
        #divide whole error into different part and distribute it depending on weight
        hidden_errors= numpy.dot(self.Who.T,output_errors)
        #renew weight matrix layer hidden to output by caculating degradient
        self.Who+=self.lr*numpy.dot(-final_outputs+targets,numpy.transpose(hidden_outputs))
        self.b2+=self.lr*(-final_outputs+targets)
       #renew weight matrix layer input to hidden by caculating degradient
        #self.Wih+=self.lr*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),numpy.transpose(inputs))
        self.Wih+=self.lr*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),numpy.transpose(inputs))
        self.b1+=self.lr*hidden_errors*hidden_outputs*(1.0-hidden_outputs)
        #self.Who+=self.lr*numpy.dot(output_errors*final_outputs*(1.0-final_outputs)*(targets*final_outputs-targets*final_outputs*final_outputs-1.0+targets)/(1.0-final_outputs),numpy.transpose(hidden_outputs))
        #self.Wih+=self.lr*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs)*(targets*hidden_outputs-targets*hidden_outputs*hidden_outputs-1.0+targets)/(1.0-hidden_outputs),numpy.transpose(inputs))
        #(targets*final_outputs-targets*final_outputs*final_outputs-1.0+targets)/(1.0-final_outputs)*
    
#Block of query() 
#refers to this block accepts the input of NN and gives feedback 
    def query(self,input_list):
#ndmin refers to min dim=2
#T refers to transpose
        inputs=numpy.array(input_list,ndmin=2).T
# Wih·I=X1
        hidden_inputs=numpy.dot(self.Wih,inputs)+self.b1
#hidden_output=sigmoid(X1)=sigmoid(Wih·I)
        hidden_outputs=self.activation_function_sigmoid(hidden_inputs)
#Who·O=X2
        final_input=numpy.dot(self.Who,hidden_outputs)+self.b2
#final_output=sigmoid(X2)=sigmoid(Who·X2)
        final_outputs=self.activation_function_softmax(final_input)
        return (final_outputs)
    def giveBackLoss(self, input_list, targets_list):
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.Wih,inputs)+self.b1
        hidden_outputs=self.activation_function_sigmoid(hidden_inputs)
        final_inputs = numpy.dot(self.Who,hidden_outputs)+self.b2
        final_outputs=self.activation_function_softmax(final_inputs)
        output_errors=-targets*numpy.log(final_outputs)
        hidden_errors= numpy.dot(self.Who.T,output_errors)
        print(sum(output_errors))


# In[33]:


#input data of nodes
input_nodes=784
hidden_nodes=900
output_nodes=10
# set learning rate 
learning_rate = 0.01
#create instance of NN(neural network)

training_data_file = open('C:/Users/32665/Desktop/bachelor_design/dataset/mnist_train.csv','r')
training_data_list=training_data_file.readlines()
training_data_file.close()
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


# In[34]:


epochs=1
flow=0
for i in range(epochs):
    for record in training_data_list:
        all_values=record.split(',')
        all_values=numpy.asfarray(all_values)
        inputs=numpy.asfarray(all_values[1:])/255.0*0.99+0.01
        #inputs=numpy.asfarray(all_values[1:])
        #image=inputs.reshape(28,28)
        #ret, binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
        #inputs=binary/255.0*0.99+0.01
        inputs=inputs.reshape(784)
        targets=numpy.zeros(output_nodes)+0.0001
        targets[int(all_values[0])]=0.992
        n.train(inputs,targets)
        flow+=1
        if flow%1000==0:
            n.giveBackLoss(inputs,targets)
        """
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        """
        #rotate sample set 10 degree
        #scaled_input=inputs
       # input_plus10_img=scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28),10,cval=0.01,reshape=False)
        #input_plus10=input_minus10_img.reshape(784)
       # input_minus10_img=scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28),-10,cval=0.01,reshape=False)  
        #input_minus10=input_minus10_img.reshape(784)
     
        #n.train(input_minus10,targets)
      #  n.train(input_plus10,targets)


# In[35]:


test_data_file=open('C:/Users/32665/Desktop/bachelor_design/dataset/mnist_test.csv','r')
test_data_list=test_data_file.readlines()
test_data_file.close()


# In[36]:


#set up a list named scoreCard to caculate time successfully recognized 
scoreCard=[]
false_number=[0,0,0,0,0,0,0,0,0,0]
for record in test_data_list:
    all_values=record.split(',')
    #get lable of number
    correct_label=int(all_values[0])
    #print(correct_label,"correct label")
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #inputs=(numpy.asfarray(all_values[1:]))
    #image=inputs.reshape(28,28)
    #ret, binary = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    #inputs=binary/255.0*0.99+0.1
    #inputs=inputs.reshape(784)
    outputs=n.query(inputs)
    #print(outputs)
    #send back to max number's index 
    label=numpy.argmax(outputs)
    #print(label,'networks answer')
    if(label==correct_label):
        scoreCard.append(1)
        #print(label)
    else:
        scoreCard.append(0)
        false_number[correct_label]+=1
        #cv2.imshow('',binary)
        #cv2.waitKey(0)
        pass
    pass
print(false_number)


# In[37]:


#caculate the rate of recognization
scoreCard_array=numpy.asarray(scoreCard)
#scoreCard_array.sum() tp sum the number which was successfully recognized
print("performance =",scoreCard_array.sum()/scoreCard_array.size)


# In[8]:


"""
numpy.savetxt('C:/Users/32665/Desktop/Who_new_SE.csv', n.Who, fmt='%f', delimiter = ',')
numpy.savetxt('C:/Users/32665/Desktop/Wih_new_SE.csv', n.Wih, fmt='%f', delimiter = ',')
"""


# In[9]:


#testing painting number 
image_array=cv2.imread('C:/Users/32665/Desktop/4.png')
dst = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
image_data=~dst
image_data=image_data.reshape(784)
image_data=image_data/255.0*0.99+0.01


# In[ ]:


number=list(n.query(image_data)).index(max(n.query(image_data)))

print(number)



# In[ ]:




