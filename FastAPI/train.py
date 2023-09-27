import joblib
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src/"))

from load_datasets import load_lithuanian_salary_data
from main import split_data_to_xy, create_lr_model
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.3


def main():
    data = load_lithuanian_salary_data()
    X, y = split_data_to_xy(data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE)
    lr_model = create_lr_model()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "model.joblib")


if __name__ == "__main__":
    main()
