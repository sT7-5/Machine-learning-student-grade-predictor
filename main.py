import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#split the data from student-mat.csv into an 80/20 split. 80% used for training, 20% used for testing
#first table to be used as training data and second tableto be used as testing data
data = pd.read_csv('student-mat.csv', delimiter=";")
trainData = data.iloc[:320]
testData = data.iloc[320:]


#changing the data using one-hot encoding
#converting the 'yes' or 'no' columns into boolean True or False columns
#also converts columns like Mjob who have a choice of 4 values into 4 different columns for each choice. these columns are then true or false. e.g. if Mjob is health for a row, Mjob_health = true, Mjob_other = false, Mjob_services=false, Mjob_teacher=flase for that row too.
trainDataNew = pd.get_dummies(trainData, columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], drop_first=True)

testDataNew = pd.get_dummies(testData, columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], drop_first=True)


#cleaning data (removing any rows that have null values)
trainDataNew = trainDataNew.dropna()
testDataNew = testDataNew.dropna()


#print(trainDataNew)


#seeing what columns most strongly correlate to the final grade achieved
#print(trainDataNew.corr()['G3'])


#only picking columns that had shown weak/medium to strong correlation to G3
predictor_columns = ['G1', 'G2', 'failures', 'Medu', 'goout', 'higher_yes']
#picking column we want to try and predict
target_column = 'G3'


reg = LinearRegression()


#training the ML model using the columns that had shown correlation
reg.fit(trainDataNew[predictor_columns], trainDataNew[target_column])

#getting predictions for G3 for the test data and creating a new column 'predictions' for these values
testDataNew['predictions'] = reg.predict(testDataNew[predictor_columns])
#some G3 predictions were decimal so need to round them
testDataNew['predictions'] = testDataNew['predictions'].round()
#creating a new column for the difference between the prediction and actual G3 grade
testDataNew['difference'] = (testDataNew['G3'] - testDataNew['predictions'])


#print(testDataNew['predictions'])
#print(testDataNew['G3'])


#getting how close on average ML model was to the actual grade G3
#on average we were 1.28 of a grade away
print('on average we this many grades away from the actual G3 grade:')
print(mean_absolute_error(testDataNew['G3'], testDataNew['predictions']))


#writing the testData (now with the predictions column) into a .csv file
file_path = 'newData.csv'
testDataNew.to_csv(file_path, index=False)


#creating simple graph of G2 to G3
sns.lmplot(x="G3", y="G2", data=testDataNew, fit_reg=True, ci=None) #creating graph
#shows the graph produced by sns.lmplot
plt.show() 

