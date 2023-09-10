import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

N_EPOCHS = 100


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def create_torch_lr_model_and_show_loss(X_data, y_data):
    x_train = Variable(torch.Tensor(X_data["profesija"].to_numpy()))
    y_train = Variable(torch.Tensor(y_data.to_numpy()))

    input_size = len(x_train)
    output_size = input_size
    model = LinearRegressionModel(input_size, output_size)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.000001)

    # Train the model
    for epoch in range(N_EPOCHS):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")
