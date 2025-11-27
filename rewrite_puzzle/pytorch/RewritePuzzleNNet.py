"""
PyTorch Neural Network for Rewrite Puzzle Game.
Uses a fully connected network since the state is 1D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewritePuzzleNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_size = game.getBoardSize()[0]  # 1D board
        self.action_size = game.getActionSize()
        self.args = args

        super(RewritePuzzleNNet, self).__init__()
        
        # Fully connected layers for 1D input
        self.fc1 = nn.Linear(self.board_size, args.num_channels)
        self.fc_bn1 = nn.BatchNorm1d(args.num_channels)
        
        self.fc2 = nn.Linear(args.num_channels, args.num_channels)
        self.fc_bn2 = nn.BatchNorm1d(args.num_channels)
        
        self.fc3 = nn.Linear(args.num_channels, args.num_channels)
        self.fc_bn3 = nn.BatchNorm1d(args.num_channels)
        
        self.fc4 = nn.Linear(args.num_channels, args.num_channels)
        self.fc_bn4 = nn.BatchNorm1d(args.num_channels)

        self.fc5 = nn.Linear(args.num_channels, 512)
        self.fc_bn5 = nn.BatchNorm1d(512)

        self.fc6 = nn.Linear(512, 256)
        self.fc_bn6 = nn.BatchNorm1d(256)

        self.fc_pi = nn.Linear(256, self.action_size)  # Policy head
        self.fc_v = nn.Linear(256, 1)  # Value head

    def forward(self, s):
        """
        Args:
            s: batch_size x board_size (1D tensor)
        """
        s = s.view(-1, self.board_size)  # batch_size x board_size
        
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn3(self.fc3(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn4(self.fc4(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn5(self.fc5(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn6(self.fc6(s))), p=self.args.dropout, training=self.training)

        pi = self.fc_pi(s)  # batch_size x action_size
        v = self.fc_v(s)    # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

