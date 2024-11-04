import sklearn
import pandas as pd

df = pd.read_csv(r"D:\Datasets\Machine Learning\tip.csv")

df.head()

df.info()

from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
df['Gender'] = Encoder.fit_transform(df['sex'])
df.drop(columns='sex',inplace=True)
df['smoker'] = Encoder.fit_transform(df['smoker'])
df['day'] = Encoder.fit_transform(df['day'])
df['time'] = Encoder.fit_transform(df['time'])

df.info()

X = df.drop(columns='smoker')
y = df['smoker']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                            test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier()

tree_classifier.fit(X_train, y_train)

y_pred = tree_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
