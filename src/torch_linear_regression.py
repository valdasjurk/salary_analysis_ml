import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import logging

N_EPOCHS = 100
CAT_FEATURES = ["lytis", "amzius", "issilavinimas"]
LOGGER_FILENAME = "logger.log"

logging.basicConfig(
    filename=LOGGER_FILENAME,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def preprocess_data(X_data):
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_cat_data = pd.DataFrame(ohe.fit_transform(X_data[CAT_FEATURES]).toarray())
    X_num_data = X_data.drop(CAT_FEATURES, axis=1)
    X_data_preprocessed = pd.concat(
        [X_num_data.reset_index(drop=True), X_cat_data.reset_index(drop=True)],
        axis=1,
    )
    return X_data_preprocessed


def convert_data_to_torch(X_train, y_train, X_test, y_test):
    X_train = preprocess_data(X_train)
    X_train = Variable(torch.Tensor(X_train.to_numpy()))
    y_train = [[i] for i in y_train]
    y_train = Variable(torch.Tensor(y_train))

    X_test = preprocess_data(X_test)
    X_test = Variable(torch.Tensor(X_test.to_numpy()))
    y_test = [[i] for i in y_test]
    y_test = Variable(torch.Tensor(y_test))
    return X_train, y_train, X_test, y_test


def create_torch_lr_model_and_show_loss(X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = convert_data_to_torch(
        X_train, y_train, X_test, y_test
    )
    model = LinearRegressionModel(X_train.shape[1], y_train.shape[1])
    # Define loss and optimizer
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=0.000001)
    loss_values = []
    # Train the model
    for epoch in range(N_EPOCHS):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")
    # evaluate model at end of epoch
    with torch.no_grad():
        y_pred = model(X_test)
        mse = criterion(y_pred, y_test)
        logging.info(f"Torch MSE: {mse}")
    return mse, loss_values
