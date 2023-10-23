# Import necessary libraries
import pandas as pd            # Import pandas for data manipulation and analysis
import numpy as np             # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for creating visualizations
# %matplotlib inline            # This magic command is commented out but is used to display plots inline in Jupyter Notebook
import seaborn as sns          # Import seaborn for statistical visualizations

# Load the dataset from a CSV file named "insurance3r2.csv"
data = pd.read_csv("insurance3r2.csv")

# Display the first few rows of the dataset
data.head()

# Display information about the dataset, including data types and missing values
data.info()

# Generate descriptive statistics of the dataset
data.describe()

# Remove rows with missing values (NaN)
data = data.dropna()

# Create a bar plot to show the distribution of the target variable 'insuranceclaim'
plt.title('Class Distributions \n (0: No Claim || 1: Claim)', fontsize=14)
sns.set(style="darkgrid")  # Set the style of seaborn plots to "darkgrid"
sns.countplot(data['insuranceclaim'])  # Create a count plot of 'insuranceclaim'
plt.grid()  # Add grid lines to the plot
plt.show()  # Display the plot

# Compute the correlation matrix between numerical features
corr = data.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True)  # Annotate the heatmap with correlation values
plt.show()

# Drop the 'region' column from the dataset
data = data.drop('region', axis=1)

# Display the first few rows of the modified dataset
data.head()

# Create a bar plot showing the relationship between 'age' and 'charges'
plt.figure(figsize=(16, 8))
sns.barplot(x='age', y='charges', data=data)
plt.title("Age vs Charges")

# Create a bar plot showing the relationship between 'sex' and 'charges'
plt.figure(figsize=(6, 6))
sns.barplot(x='sex', y='charges', data=data)
plt.title('sex vs charges')

# Create a bar plot showing the relationship between 'children' and 'charges'
plt.figure(figsize=(12, 8))
sns.barplot(x='children', y='charges', data=data)
plt.title('children vs charges')

# Create a bar plot showing the relationship between 'smoker' and 'charges'
plt.figure(figsize=(6, 6))
sns.barplot(x='smoker', y='charges', data=data)
plt.title('smoker vs charges')

# Extract the features (input variables) by selecting all columns except the last one
X = data.iloc[:, :-1]
X.head()

# Display the shape of the feature matrix
X.shape

# Extract the target variable by selecting the last column
Y = data.iloc[:, -1]
Y.head()

# Display the shape of the target variable
Y.shape

# Import the train_test_split function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Save the original dataset without missing values to 'finaldata.csv'
data.to_csv('finaldata.csv')

# Save the testing dataset to 'testing.csv'
X_test.to_csv('testing.csv')

# Import StandardScaler from sklearn.preprocessing to scale the features
from sklearn.preprocessing import StandardScaler

# Uncomment the following lines to scale the features using StandardScaler
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_train = pd.DataFrame(X_train, columns=X_test.columns)
# X_test = ss.fit_transform(X_test)
# X_test = pd.DataFrame(X_test, columns=X_train.columns)

# Display the first few rows of the scaled training features
# X_train.head()

# Import necessary libraries for model evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import RandomForestClassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier instance
rf = RandomForestClassifier()

# Fit the RandomForestClassifier on the training data
rf.fit(X_train, y_train)

# Predict the target values on the testing data
ypred = rf.predict(X_test)

# Display the confusion matrix
print(confusion_matrix(y_test, ypred))

# Import cross_val_score from sklearn.model_selection
from sklearn.model_selection import cross_val_score

# Perform cross-validation on the RandomForestClassifier
acc = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10)

# Calculate the mean and standard deviation of cross-validation accuracy
acc_mean = acc.mean()
acc_std = acc.std()

# Import pickle for model serialization
import pickle

# Save the trained model to disk using pickle
pickle.dump(rf, open('model.pkl', 'wb'))

# Load the saved model from disk using pickle
model = pickle.load(open('model.pkl', 'rb'))
