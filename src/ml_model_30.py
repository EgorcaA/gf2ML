import torch
import torch.nn as nn
import time

class MLSolver(nn.Module):
    def __init__(self, input_size=42, output_size=30):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            # Your network structure comes here
            nn.Linear(input_size, 64),  # First hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(64, 128),           # Second hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(128, 64),           # Third hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(64, output_size)    # Output layer
        )    
        #     nn.Linear(784, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ELU(),
        #     nn.Linear(32,10),
        #     nn.Linear(input_shape, num_classes)
        # )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3) 
        
    def forward(self, inp):       
        out = self.model(inp)
        # print(out)
        return out

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.load_state_dict(state['state_dict'])
        self.opt.load_state_dict(state['optimizer'])
        print('model loaded from %s' % checkpoint_path)
