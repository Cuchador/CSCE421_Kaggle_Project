import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(x_path) -> pd.DataFrame:
    # Your code here
    return pd.read_csv(x_path, low_memory=False)


def split_data(x, y, test_split=0.2):
    # Your code here
    return train_test_split(x, y, test_size=test_split, random_state=42)


def get_unique_patientids(train_x: pd.DataFrame) -> np.ndarray:
    return pd.unique(train_x['patientunitstayid'])


def get_admission_values(x, train_x, column_order):
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
def reformat_x(train_x: pd.DataFrame) -> pd.DataFrame:
    # Grab unique patientids
    patient_ids = get_unique_patientids(train_x)

    # Create a dataframe with the template we want and put patient ids in it
    column_order = ["patientunitstayid", "unitvisitnumber", "admissionweight", "admissionheight",
                            "age", "ethnicity", "gender"]
    x = pd.DataFrame({'patientunitstayid': patient_ids})
    
    #get the admission values
    grouped_x = get_admission_values(x, train_x, column_order)
    
    #grab the lab data from train_x and put it into our grouped data
    reformatted_x = group_lab_averages(grouped_x, get_lab_averages(train_x))
    
    # Return the resulting dataframe
    #reformatted_x.to_csv("test.csv")
    return reformatted_x


def fix_ages(feature_columns) -> pd.DataFrame:
    feature_columns["age"] = feature_columns["age"].replace({"> 89": 90})
    return feature_columns

def fill_missing_num_values(num_columns: pd.DataFrame):
    num_columns.ffill()
    # num_columns['unitvisitnumber'] = num_columns['unitvisitnumber'].fillna(num_columns['unitvisitnumber'].median())
    # num_columns['admissionweight'] = num_columns['admissionweight'].fillna(num_columns['admissionweight'].mean())
    # num_columns['admissionheight'] = num_columns['admissionheight'].fillna(num_columns['admissionheight'].mean())
    # num_columns['age'] = num_columns['age'].fillna(num_columns['age'].mean())
    # num_columns['glucose_avg'] = num_columns['glucose_avg'].fillna(num_columns['glucose_avg']).mean()
    # num_columns['ph_avg'] = num_columns['ph_avg'].fillna(num_columns['ph_avg'].mean())
    return num_columns

def force_numeric(feature_columns: pd.DataFrame) -> pd.DataFrame:
    feature_columns['unitvisitnumber'] = pd.to_numeric(feature_columns['unitvisitnumber'], errors='coerce', downcast='float')
    feature_columns['admissionweight'] = pd.to_numeric(feature_columns['admissionweight'], errors='coerce', downcast='float')
    feature_columns['admissionheight'] = pd.to_numeric(feature_columns['admissionheight'], errors='coerce', downcast='float')
    feature_columns['age'] = pd.to_numeric(feature_columns['age'], errors='coerce', downcast='float')
    return feature_columns


def preprocess_x_y(reformatted_x: pd.DataFrame, y: pd.DataFrame):
    #x = reformatted_x.dropna()
    x = reformatted_x
    #extract features and labels
    #label_column = pd.merge(x, y, on='patientunitstayid')['hospitaldischargestatus']
    feature_columns = x.drop("patientunitstayid", axis='columns')
    feature_columns = fix_ages(feature_columns)
    
    #force numeric columns to be numeric
    feature_columns = force_numeric(feature_columns)
    
    
    #select categorical vs numerical
    nan_columns = feature_columns.select_dtypes(exclude=['int', 'float'])
    num_columns = feature_columns.select_dtypes(include=['int', 'float'])
    
    #fill in missing values with middle values
    #this step is necessary so that we do not have to drop more rows
    num_columns = fill_missing_num_values(num_columns)
    
    #transform categorical data
    #dummies = pd.get_dummies(nan_columns)
    #features = pd.concat([dummies, num_columns], axis=1)
    
    num_columns.to_csv("output.csv")
    #return features, label_column
    