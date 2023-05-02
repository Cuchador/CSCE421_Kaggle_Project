import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from data import load_data, split_data, reformat_x, preprocess_x_y
from model import Model
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load the data
    reformatted_x = reformat_x(load_data("data_files/train_x.csv"))
    y = load_data("data_files/train_y.csv")
    return
    preprocessed_x = preprocess_x_y(reformatted_x, y)
    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y = split_data(preprocessed_x, y, test_split=0.2)

    # Initialize and train the model
    model = RandomForestClassifier()  # Add arguments as needed
    model.fit(train_x, train_y)

    # Make predictions on the test set
    predictions = model.predict(test_x)[:, 1]

    # Evaluate the model using accuracy score
    accuracy = roc_auc_score(test_y["hospitaldischargestatus"], predictions)
    print("Accuracy Score:", accuracy)

    #### Your Code Here ####


if __name__ == "__main__":
    main()
