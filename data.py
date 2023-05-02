import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(x_path):
    # Your code here
    return pd.read_csv(x_path, low_memory=False)


def split_data(x, y, test_split=0.2):
    # Your code here
    return train_test_split(x, y, test_size=test_split, random_state=42)


def get_unique_patientids(train_x: pd.DataFrame) -> np.ndarray:
    return pd.unique(train_x['patientunitstayid'])


def get_addmission_values(x, train_x, column_order):
    # Merge the dataframes on the column 'patientunitstayid'
    x = pd.merge(x, train_x, on='patientunitstayid', how='left')

    # Group the data by the patient id and concatenate the data points
    # gets the dataframe to grab data in the desired order
    grouped_x = x.groupby('patientunitstayid').agg(lambda x: ', '.join(x.dropna().astype(str)))
    grouped_x = grouped_x.reset_index()

    # Merge with the original x dataframe
    grouped_x = pd.merge(x[['patientunitstayid']].drop_duplicates(), grouped_x, on='patientunitstayid')

    # Reorder the columns based on the given column_order
    grouped_x = grouped_x.reindex(columns=column_order)
    
    return grouped_x

def get_lab_averages(train_x):
    # Get average lab values for each patient
    lab_data = train_x.loc[train_x['labname'].isin(['pH', 'glucose']), ['patientunitstayid', 'labname', 'labresult']]
    lab_data = lab_data.groupby(['patientunitstayid', 'labname'])['labresult'].mean().unstack()
    return lab_data

def group_lab_averages(grouped_x, lab_data):
    # Merge lab data into the dataframe
    grouped_x = pd.merge(grouped_x, lab_data, on='patientunitstayid', how='left')
    grouped_x = grouped_x.rename(columns={'glucose': 'glucose_avg', 'pH': 'ph_avg'})
    return grouped_x

# takes in train_x read in as a dataframe, returns a preprocessed dataframe


def reformat_x(train_x) -> pd.DataFrame:
    # Grab unique patientids
    patient_ids = get_unique_patientids(train_x)

    # Create a dataframe with the template we want and put patient ids in it
    column_order = ["patientunitstayid", "unitvisitnumber", "admissionweight", "admissionheight",
                            "age", "ethnicity", "gender"]
    x = pd.DataFrame({'patientunitstayid': patient_ids})
    
    #get the admission values
    grouped_x = get_addmission_values(x, train_x, column_order)
    
    #grab the lab data from train_x and put it into our grouped data
    reformatted_x = group_lab_averages(grouped_x, get_lab_averages(train_x))
    
    # Return the resulting dataframe
    #reformatted_x.to_csv("test.csv")

    #one hot encode gender

    return reformatted_x
    