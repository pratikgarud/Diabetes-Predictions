import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')
feature_col = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_col].astype('int')
df['Outcome'] = df['Outcome'].replace({0:'Diabetes', 1:'Not Diabetes'})
y = df['Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=142, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(model.score(X_test,y_test)*100)

import pickle
with open('Diabetes_Model.pkl','wb') as f:
    pickle.dump(model,f)