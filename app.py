import pandas as pd #library for manipulating the datasets
import numpy as np #this library will be used to deal with NaN values

df_present = pd.read_csv('air_system_present_year.csv') #reading the csv

df_previous = pd.read_csv('air_system_previous_years.csv') #reading the csv

df = df_present.merge(df_previous, how='outer') #merging the dataframes

df.replace('na', np.nan, inplace=True)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.astype(float))

ten_percent = 0.1*df.shape[0]

dropped_columns = []
for col in df.columns:
    if (df[col].eq(0).sum() > ten_percent or
        df[col].isna().sum() > ten_percent or
        (df[col].eq(0).sum() + df[col].isna().sum()) > ten_percent):
        df.drop(col, axis=1, inplace=True)
        dropped_columns.append(col)


df.iloc[:,1:] = df.iloc[:,1:].apply(lambda col: col.fillna(col.median()))

zero_rows = (df == 0).all(axis=1) # rows filled with zeros

df = df[~zero_rows]

df['class'] = df['class'].map({'pos': 1, 'neg': 0}) #enconding label column

# Remove label column from the dataset
y = df['class']
X = df.drop(columns=['class'])

from imblearn.under_sampling import RandomUnderSampler # balancing the data set

sampling_strategy = 0.5 #50% 1, 50% 0
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X, y = rus.fit_resample(X, y)

from sklearn.model_selection import train_test_split #dividing the dataset in train, validation and test sets
from sklearn.preprocessing import StandardScaler #rescaling the dataset
from sklearn.decomposition import PCA # principal component analysis

# Spliting the dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Standardazing the scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Applying PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

# Explained variance
variance_diff = np.diff(pca.explained_variance_)

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

#from sklearn.metrics import classification_report, confusion_matrix #metrics
from sklearn.model_selection import GridSearchCV #hyperparameters tuning

def gridSearch(clf, grid):

    grid_search = GridSearchCV(estimator=clf, param_grid=grid,
                                       cv=5, scoring='recall') # choosing the best hyperparameters in a fast and effective way
    grid_search.fit(X_train_pca, y_train)

    # Model with the best hyperparameters
    best_clf = grid_search.best_estimator_

    # Tresting the model on validation set
    #y_pred_val = best_clf.predict(X_val_pca)

    # Testing the model on test set
    #y_pred_test = best_clf.predict(X_test_pca)
    return best_clf

from sklearn.neighbors import KNeighborsClassifier

# K Nearest Neightbors
clf = KNeighborsClassifier()

# Hyperparameters
param_grid = {
    'n_neighbors': [3, 5, 10],  # number of neighbors
    'weights': ['uniform', 'distance'],  # neighbors weight
    'metric': ['euclidean', 'manhattan']  # distance metric
}

best_clf = gridSearch(clf=clf, grid=param_grid)

def predict(file): #encapsulating the prediction model
    df = pd.read_csv(file, sep=";")
   # df = file.copy()
    df.drop(columns=dropped_columns, inplace=True)
    df.replace('na', np.nan, inplace=True)
    df = df.apply(lambda x: x.astype(float))
    df = df.apply(lambda col: col.fillna(col.median()))
    df_scaled = scaler.transform(df)
    df_pca = pca.transform(df_scaled)
    prediction = best_clf.predict(df_pca)
    prediction = pd.Series(prediction)
    return prediction

# these libraries will be used to create an interface for new predictions
import gradio as gr
from gradio import File, Text

title = "Prediction model"
description = """
Best practices guide for the model interface:
- Load a csv file with 170 features columns and no label column
- The csv separator must be semicolon

Load your csv and predict wether the truck will fail or not:"""

interface = gr.Interface(
    fn=predict,
    inputs=[File(label="Upload CSV")],
    outputs=[Text(label="Predictions output")],
    title=title,
    description=description
)
if __name__ == "__main__":
    interface.launch()