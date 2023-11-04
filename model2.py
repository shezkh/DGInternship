# importing libraries
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


dataset = pd.read_csv("Salary_dataset.csv", index_col=(0))
dataset.info()
print(dataset.head())

# X is the independent variable and y is the dependent variable
X = dataset[["YearsExperience"]]
y = dataset[["Salary"]]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting the linear regression model
regressor.fit(X, y)

#saving the linear regression model to disk
pickle.dump(regressor, open('model2.pkl', 'wb'))

#loading model to compare the results
model = pickle.load(open('model2.pkl', 'rb'))

#predicting salary for 5 years of experience
print(model.predict([[5]]))
