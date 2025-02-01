from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim, device):
        super(VanillaRNN, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        # w_hx * xt + bh
        self.hx = nn.Linear(input_dim, hidden_dim, bias=True)
        # w_hh * h_(t-1)
        self.hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # o_t = w_ph * h_t + b_o
        self.ho = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        batch_size, input_length = x.size(0), x.size(1)
        h_last = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for t in range(input_length):
            x_cur = x[:, t, :]
            h_last = torch.tanh(self.hx(x_cur) + self.hh(h_last))

        out = torch.softmax(self.ho(h_last), dim=-1)
        return out
