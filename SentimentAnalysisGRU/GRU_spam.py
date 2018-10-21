#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:55:04 2018

@author: kaushik
"""

import numpy as np
# import matplotlib.pyplot as plt 
import dataset
import torch
from torch.autograd import Variable
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, vocab_size, input_feature_size, hidden_size, output_size, lr=0.001):
        super(MyRNN, self).__init__()
        torch.manual_seed(10)
        self.hidden_size = hidden_size
        self.vs = vocab_size
        self.embedding = nn.Embedding(vocab_size, input_feature_size)
        self.gru = nn.GRU(input_feature_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.hidden = 0
        self.lr = lr
        

    def forward(self, ip1):
        embed = self.embedding
        ip = embed(ip1)
        hidden_layer = self.hidden
        output, hidden_layer = self.gru(ip, hidden_layer)
        self.hidden = hidden_layer
        lin_out = self.linear(output[-1,:,:])
        return lin_out
    
    def train(self, trainloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0
            for batch in trainloader:
                # get the inputs
                inputs, labels = batch.sentence, batch.label
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                self.hidden = Variable(self.initHidden(inputs))
                
                # compute forward pass
                outputs = self.forward(inputs)
                
                # get loss function
                loss = criterion(outputs, labels)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
            print(running_loss)
        
    def predict(self, testloader):
        correct = 0
        total = 0
        all_predicted = []
        for batch in testloader:
                # get the inputs
                inputs, labels = batch.sentence, batch.label
                self.hidden = Variable(self.initHidden(inputs))
                outputs = self.forward(inputs)
                soft_out = self.softmax(outputs)
                _, prediction = torch.max(soft_out,1)
                
                total += labels.size(0)
                # print("Total = ", total)
                x = (prediction == labels).sum()
                # print("x = ", x)
                correct += x.data.type(torch.DoubleTensor)
                # print("correct", correct.data.numpy())
                all_predicted += prediction.data.numpy().tolist()

        accuracy = (100.0 * float(correct)/float(total))
        model_filename = dataset_category + "_" + str(accuracy) + "_" + "model.pt"
        torch.save(rnn, model_filename)

        return accuracy
                

    def initHidden(self,ip):
        return torch.zeros(1, ip.data.numpy().shape[1], self.hidden_size)
    
dataset_category = "Spam"
model_filename = dataset_category + "model"
train_iter, test_iter, vocab_size = dataset.load_dataset(dataset_category, batch_size=20)
epochs = 16
hs = 20
ifs = 150
lr = 0.001
print("Hidden Size = ", hs)
print("input_feature_size = ", ifs)
print("lr = ", lr)
torch.manual_seed(10)
rnn = MyRNN(vocab_size, input_feature_size=ifs, hidden_size=hs, output_size=2, lr=lr)

##############################################################
### PERFORM TRAINING - STORES THE MODEL WITH HIGHEST ACCURACY
##############################################################

for i in range(epochs):
    rnn.train(train_iter)
    accuracy = rnn.predict(test_iter)
    if i==0:
       max_acc = np.copy(accuracy) 
       torch.save(rnn, model_filename+str(i)+".pt")
       continue
    else:
	if accuracy > max_acc:
	   torch.save(rnn, model_filename+str(i)+".pt") 
	   max_acc = np.copy(accuracy)     
       
torch.save(rnn, model_filename+str(int(np.ceil(max_acc)))+".pt")

###############################################################
### PERFORM PREDICTION - PREDICT FROM PREVIOUSLY TRAINED MODEL
###############################################################
       
rnn = torch.load('./models/spammodel95.pt')
accuracy = rnn.predict(test_iter)
print('Accuracy of the network is {}'.format(accuracy))
