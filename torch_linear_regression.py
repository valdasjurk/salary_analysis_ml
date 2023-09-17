import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

N_EPOCHS = 100
CAT_FEATURES = ["lytis", "amzius", "issilavinimas"]


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def prepare_data(X_data):
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_cat_data = pd.DataFrame(ohe.fit_transform(X_data[CAT_FEATURES]).toarray())
    X_num_data = X_data.drop(["lytis", "amzius", "issilavinimas"], axis=1)
    X_data_preprocessed = pd.concat(
        [X_num_data.reset_index(drop=True), X_cat_data.reset_index(drop=True)],
        axis=1,
    )
    return X_data_preprocessed


def create_torch_lr_model_and_show_loss(X_train, y_train, X_test, y_test):
    X_train = prepare_data(X_train)
    x_train = Variable(torch.Tensor(X_train.to_numpy()))
    y_train = [[i] for i in y_train]
    y_train = Variable(torch.Tensor(y_train))

    X_test = prepare_data(X_test)
    X_test = Variable(torch.Tensor(X_test.to_numpy()))
    y_test = [[i] for i in y_test]
    y_test = Variable(torch.Tensor(y_test))

    model = LinearRegressionModel(x_train.shape[1], y_train.shape[1])
    # Define loss and optimizer
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=0.000001)
    # loss_values = []
    # Train the model
    for epoch in range(N_EPOCHS):
        # Forward pass
        outputs = model(x_train)
        # print("outputs: ", outputs)
        loss = criterion(outputs, y_train)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_values.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")
    # plt.plot(loss_values)
    # plt.show()
    # evaluate model at end of epoch
    with torch.no_grad():
        y_pred = model(X_test)
        acc = (y_pred == y_test).float().mean()
        acc = float(acc)
        print(f"accuracy {acc}")
