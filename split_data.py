from sklearn.model_selection import train_test_split
import pandas as pd


# take csv file containing image names and genre classifications in full dataset
# use sklearn to randomly divide the set into 70% training data and 30% validation data
# save these two datasets as csv files

data_loc = './../project data/'

names_df = pd.read_csv(data_loc + "names.csv")

names_df = names_df.drop(columns= ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'])

image_names = names_df['Image Name']
genres = names_df['Genre']

X_train, X_validate, y_train, y_validate = train_test_split(image_names, genres, train_size=0.7)

train_dict = {'Image Name': X_train, 'Genre': y_train}
train_data = pd.DataFrame(train_dict)

validate_dict = {'Image Name': X_validate, 'Genre': y_validate}
validate_data = pd.DataFrame(validate_dict)

train_data.to_csv(data_loc + "train_data.csv")
validate_data.to_csv(data_loc + "validate_data.csv")