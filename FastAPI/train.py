import joblib
from src.main import load_lithuanian_salary_data, split_data_to_xy, create_lr_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

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
