import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

data1 = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data1.loc[:,'AgeNotNull'] = data1.Age
data1.loc[data1.Age.isnull(),'AgeNotNull'] = data1.loc[:,'Age'].mean()

data1.loc[data1.Embarked == 'Q','Port'] = 0
data1.loc[data1.Embarked == 'S','Port'] = 1
data1.loc[data1.Embarked == 'C','Port'] = 2
data1.loc[data1.Embarked.isnull(),'Port'] = 1

data1.loc[data1.Sex == 'male','Gender'] = 0
data1.loc[data1.Sex == 'female','Gender'] = 1

data1 = data1.drop(['Embarked','Sex','Age'],axis=1)


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(np.array(data1)[:,1:], np.array(data1)[:,0] )


test_data1 = test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test_data1.loc[:,'AgeNotNull'] = test_data1.Age
test_data1.loc[test_data1.Age.isnull(),'AgeNotNull'] = test_data1.loc[:,'Age'].mean()

test_data1.loc[test_data1.Embarked == 'Q','Port'] = 0
test_data1.loc[test_data1.Embarked == 'S','Port'] = 1
test_data1.loc[test_data1.Embarked == 'C','Port'] = 2
test_data1.loc[test_data1.Embarked.isnull(),'Port'] = 1

test_data1.loc[test_data1.Sex == 'male','Gender'] = 0
test_data1.loc[test_data1.Sex == 'female','Gender'] = 1

test_data1 = test_data1.drop(['Embarked','Sex','Age'],axis=1)
test_data1.loc[test_data1.Fare.isnull(),'Fare'] = test_data1.loc[:,'Fare'].mean()	



output = forest.predict(np.array(test_data1)[:,1:])
output_df = pd.DataFrame( {'PassengerId' : test_data.loc[:,'PassengerId'], 'Survived': output.astype(int)})

print output_df
output_df.to_csv('result.csv',index=False)

