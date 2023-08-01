import matplotlib
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
# Stochastic Gradient Descent
#from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression

modele = LinearRegression()
df = pd.read_csv("globalmeantemp.csv")
print(df.keys())
x = df["Year"].values.reshape(-1,1)
y = df["Nasa GISS"]

modele.fit(y=y,X=x)
print("intercept = ", modele.intercept_)
print("slope = ", modele.coef_)
print("warming per decade = ", modele.coef_[0]*10, "C/decade")
print("warming per century = ", modele.coef_[0]*100, "C/century")

print("temperature in 2000 = ", modele.predict([[2000]]), " C")
print("temperature in 2050 = ", modele.predict([[2050]]), " C")
print("temperature in 2100 = ", modele.predict([[2100]]), " C")
r2 = modele.score(X=x,y=y)
print(r2)


X_modele = x
Y_modele = modele.predict(x)

matplotlib.rcParams['font.family'] = 'serif'
plt.scatter(x,y,label = 'données de départ') 
plt.plot(X_modele,Y_modele, '-r',label = 'droite de régression')
plt.title("Regression linéaire univariée")
plt.ylim(ymin=0)
plt.xlabel('année')
plt.ylabel('température (C)')
plt.legend(loc='lower right', frameon=False)
plt.savefig('linearregression.pdf')
