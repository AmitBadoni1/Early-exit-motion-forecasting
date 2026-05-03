import torch
import torch.nn as nn
import torch.nn.functional as F

class ExitHead(nn.Module):
    def __init__(self, hidden_size=128, num_modes=6, future_steps=30):
        super().__init__()
        
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.reg = nn.Linear(hidden_size, num_modes * future_steps * 2)
        self.cls = nn.Linear(hidden_size, num_modes)
        
    def forward(self, x):
        # x = [N_actor,128]

        cls = F.softmax(self.cls(x), dim=1)

        traj = self.reg(x)
        traj = traj.view(x.shape[0], self.num_modes, self.future_steps, 2)

        return cls, traj