import pandas as pd
import numpy as np

df = pd.readcsv("games.csv", sep=",")
'''
values = np.array(values[:1000000])
columns = universal_cols
index = range(len(values))
print "values ready!"
#print master_df['BT']
X = values[:,7:]
y = values[:,3]
y[y=='w']=1.
y[y=='b']=0.
y[(y!='w')&(y!='b')]=.5
y=y.astype(float)
for row in X:
        row[row=='w'] = 1.
        row[row=='0'] = .5
        row[row=='b'] = 0.
        row = row.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print "accuracy:", accuracy_score(y_test, y_pred)
'''
