import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torchvision.models as models
from torchvision import transforms, utils



class MLP(nn.Module):  
  def __init__(self, num_classes, rnn_layers, hidden_size, fc_size):
    super(MLP, self).__init__()
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.fc_size = fc_size
    self.apply(self._init_weights)
    self.fc_pre= nn.Sequential(
                            nn.Linear(138, fc_size),
                           nn.ReLU())                          
    
    self.rnn = nn.GRU(input_size = fc_size,
                hidden_size = hidden_size,
                num_layers = rnn_layers,
                batch_first = True)
    self.fc = nn.Linear(hidden_size, num_classes)

  
  def init_hidden(self, num_layers, batch_size):
             return (torch.zeros(num_layers, batch_size, self.hidden_size).cuda(),
                     torch.zeros(num_layers, batch_size, self.hidden_size).cuda())
  
   
  def forward(self, inputs, hidden=None, steps=0):
        length = len(inputs)
        fs = torch.zeros(inputs[0].size(0), length, self.rnn.input_size).cuda()
        for i in range(length):
            f = inputs[i]
            
            f = f.view(f.size(0), -1)
            f = self.fc_pre(f)
            fs[:, i, :] = f       
        outputs, hidden = self.rnn(fs, hidden)
        outputs = self.fc(outputs)
        return outputs

  def _init_weights(self, module):
         if isinstance(module, nn.Linear):
             module.weight.data.uniform_(mean=0.0, std=1.0)
             if module.bias is not None:
                 module.bias.data.zero_()

