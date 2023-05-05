import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, roc_curve
from data import load_data, split_data, reformat_x, preprocess
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from IPython.display import display
import matplotlib.pyplot as plt

def main():
    # Load the data
    # reformatted_x = reformat_x(load_data("data_files/train_x.csv"))
    
    y = load_data("data_files/train_y.csv")
    x = load_data("preprocessed_x.csv")
    
    # test = reformat_x(load_data("data_files/test_x.csv"))
    #return
    
    # preprocessed_x, _ = preprocess(reformatted_x)
    # preprocessed_test, indices = preprocess(test)
    
    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y = split_data(x, y["hospitaldischargestatus"], test_split=0.2)

    # Initialize and train the model
    model = GradientBoostingClassifier(max_depth=5)  # Add arguments as needed
    model.fit(train_x, train_y)

    # Make predictions on the test set
    predictions = [i[1] for i in model.predict_proba(test_x)]
    display(train_y)
    print(model.predict_proba(test_x))

    # Evaluate the model using accuracy score
    accuracy = roc_auc_score(test_y, predictions)
    print("Accuracy Score:", accuracy)
    fpr, tpr, thresholds = roc_curve(test_y, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Display the ROC curve and AUC-ROC score
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # df = pd.DataFrame(predictions, index=[int(i) for i in indices])
    # display(df)
    # df.to_csv("output.csv")


if __name__ == "__main__":
    main()
