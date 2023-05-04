import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from data import load_data, split_data, reformat_x, preprocess
from model import Model
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

def main():
    # Load the data
    reformatted_x = reformat_x(load_data("data_files/train_x.csv"))
    y = load_data("data_files/train_y.csv")
    test = reformat_x(load_data("data_files/test_x.csv"))
    #return
    preprocessed_x, _ = preprocess(reformatted_x)
    preprocessed_test, indices = preprocess(test)
    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y = split_data(preprocessed_x, y["hospitaldischargestatus"], test_split=0.2)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=200)  # Add arguments as needed
    model.fit(train_x, train_y)

    # Make predictions on the test set
    predictions = [i[1] for i in model.predict_proba(preprocessed_test)]
    display(train_y)
    print(model.predict_proba(test_x))

    # Evaluate the model using accuracy score
    #accuracy = roc_auc_score(test_y, predictions)
    #print("Accuracy Score:", accuracy)

    df = pd.DataFrame(predictions, index=[int(i) for i in indices])
    display(df)
    df.to_csv("output.csv")


if __name__ == "__main__":
    main()
