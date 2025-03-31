from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


def train_model(no_epochs):
    best_loss= 1
    bSize = 32
    data_loaders = Data_Loaders(bSize)
    model= Action_Conditioned_FF()

    loss_function= nn.MSELoss()

    traLo=[]
    tesLo=[]

    trLoss= model.evaluate(model, data_loaders.train_loader, loss_function)
    minLoss= model.evaluate(model, data_loaders.test_loader, loss_function)
    tesLo.append(minLoss)
    traLo.append(trLoss)

    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ls= []
    minLoss= model.evaluate(model, data_loaders.test_loader, loss_function)
    ls.append(minLoss)


    for epoch_i in range(no_epochs):
        print("epoch "+ str(epoch_i))
        model.train()

        for i, data in enumerate(data_loaders.train_loader): 
            inputs, label = data['input'], data['label']


            optimizer.zero_grad()

            output= model.forward(inputs)
            loss= loss_function(output, torch.Tensor([label]))
            loss.backward()
            optimizer.step()

        trLoss= model.evaluate(model, data_loaders.train_loader, loss_function)
        minLoss= model.evaluate(model, data_loaders.test_loader, loss_function)
        print("epoch loss: " + str(minLoss))
        tesLo.append(minLoss)
        traLo.append(trLoss)

        if(minLoss < best_loss):
            print("MODEL IS SAVED")
            torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)
            best_loss = minLoss

if __name__ == '__main__':
    no_epochs = 100
    train_model(no_epochs)
