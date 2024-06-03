import torch.nn as nn
# import torch.nn.functional as F

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p, alpha):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # x = F.leaky_relu(x, negative_slope=alpha)
        x = self.relu(x)
        # x = F.dropout(x, p=p)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
