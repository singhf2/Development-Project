import quandl
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as nump
# Get stock data of amazon for training
amzn_data = quandl.get("WIKI/AMZN")

# Print the fetched data to see the current prices
print(amzn_data.head())

# Get the closely adjusted price and store it for later use
amzn_data = amzn_data[['Adj. Close']]

# Show the adjusted data to see the credibility for future predictions
print(amzn_data.head())

# Set Total days to predict the prices of amazon stock in future
total_days = 20  # A Total of 20 days have been set

# Shift the data to set the parameter according to number of days
amzn_data['Prediction'] = amzn_data[['Adj. Close']].shift(-total_days)

# Show the shifted data to check for correctness
print(amzn_data.tail())

# Convert data frame to an equivalent numpy array
train_parameter_x = nump.array(amzn_data.drop(['Prediction'], 1))

# Drop the last 20 rows to be used for prediction in future
train_parameter_x = train_parameter_x[:-total_days]
print(train_parameter_x)

# Convert data frame to an equivalent numpy array
train_parameter_y = nump.array(amzn_data['Prediction'])

# Fetch the data from the parameters except for the data of 20 days
train_parameter_y = train_parameter_y[:-total_days]
print(train_parameter_y)

# Create the Support Vector Machine for prediction
support_vector = SVR(kernel='rbf', C=1e3, gamma=0.1)

# The 80 and 20 ratio is being set to train the data
training_var, testing_var, param_train, param_test = train_test_split(train_parameter_x, train_parameter_y, test_size=0.2)

# Train the Support Vector Machine to adjust according to the provided model
support_vector.fit(training_var, param_train)

# The maximum predicted percentage will be 1
alpha = support_vector.score(testing_var, param_test)
print("Confidence Interval of Support Vector Machine is : ", alpha)

# Create a new model for linear regression
regression_model = LinearRegression()

# Train the linear regression model
regression_model.fit(training_var, param_train)

# The maximum predicted percentage will be 1
alpha_linear = regression_model.score(testing_var, param_test)
print("Linear Regression confidence is : ", alpha_linear)

# Forecast the future prices for up to 20 days
linear_var = nump.array(amzn_data.drop(['Prediction'], 1))[-total_days:]
print(linear_var)

# Show the trained model of linear regression for 20 days
linear_prediction = regression_model.predict(linear_var)
print(linear_prediction)

# Show the trained model of Support Vector machine for 20 days
svm_prediction = support_vector.predict(linear_var)
print(svm_prediction)
