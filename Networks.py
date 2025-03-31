import torch
import torch.nn as nn
import numpy
import Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        self.input_size= 6

        self.hidden_size= 200
        self.output_size= 1

        super(Action_Conditioned_FF,self).__init__()

        self.input_to_hidden= nn.Linear(self.input_size, self.hidden_size)
        self.nonlinear_ativation= nn.Sigmoid()
        self.hidden_to_output= nn.Linear(self.hidden_size,self.output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hiden= self.input_to_hidden(input)
        hiden= self.nonlinear_ativation(hiden)
        output= self.hidden_to_output(hiden)

        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        predict= []
        real= []

        for idx, s in enumerate(test_loader):
            i1= torch.from_numpy(s['input']).float() if isinstance(s['input'], numpy.ndarray) else s['input'].float()
            l1= torch.from_numpy(s['label']).float() if isinstance(s['label'], numpy.ndarray) else s['label'].float()

            y1= model(i1)
            y2= l1

            predict.append(y1)
            real.append(y2)
        
        predict= torch.stack(predict)
        real= torch.stack(real)

        rShape= real.shape

        real= real.reshape([rShape[0],1])

        loss= loss_function(predict, real)


        return loss

def main():
    model= Action_Conditioned_FF()
    

if __name__ == '__main__':
    main()
