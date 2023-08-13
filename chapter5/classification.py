import pandas as pd
df = pd.read_csv('UCI_Credit_Card.csv')
print(df.head())
print(df["default.payment.next.month"][df["default.payment.next.month"]==1].count())
print(df["default.payment.next.month"][df["default.payment.next.month"]==0].count())

x = df[["AGE", "SEX", "EDUCATION", "MARRIAGE", "LIMIT_BAL","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]]
y = df["default.payment.next.month"]
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

decision = DecisionTreeClassifier(class_weight = "balanced")
decision.fit(x, y)
print(decision.score(x, y))
#print(export_text(decision))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(class_weight = "balanced"), x, y, cv=10)
print("précision %0.2f  +/- %0.2f" % (scores.mean(), scores.std()))

x = df[["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]]
scores = cross_val_score(DecisionTreeClassifier(class_weight = "balanced"), x, y, cv=10)
print("précision %0.2f  +/- %0.2f" % (scores.mean(), scores.std()))

x = df[["SEX"]]
scores = cross_val_score(DecisionTreeClassifier(class_weight = "balanced"), x, y, cv=10)
print("précision %0.2f  +/- %0.2f" % (scores.mean(), scores.std()))
decision.fit(x, y)
print(export_text(decision))

x = df[["EDUCATION"]]
scores = cross_val_score(DecisionTreeClassifier(class_weight = "balanced"), x, y, cv=10)
print("précision %0.2f  +/- %0.2f" % (scores.mean(), scores.std()))
decision.fit(x, y)
print(export_text(decision))