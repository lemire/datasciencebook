import pandas as pd 
from sklearn.linear_model import LinearRegression

modele = LinearRegression()
df = pd.read_csv("globalmeantemp.csv")
x = df["Year"].values.reshape(-1,1)
y = df["Nasa GISS"]

modele.fit(y=y,X=x)
print("intercept = ", modele.intercept_)
print("slope = ", modele.coef_)
r2 = modele.score(X=x,y=y)
print("regression score = ", r2)

print("temperature in 2100 = ", modele.predict([[2100]]), " C")



print("== 1990 ==")

x1960 = df["Year"][df["Year"] > 1960].values.reshape(-1,1)
y1960 = df["Nasa GISS"][df["Year"] > 1960]
modele1960 = LinearRegression()
modele1960.fit(y=y1960,X=x1960)
print("intercept = ", modele1960.intercept_)
print("slope = ", modele1960.coef_)
r2 = modele1960.score(X=x,y=y)
print("regression score = ", r2)

print("temperature in 2100 = ", modele1960.predict([[2100]]), " C")

###
from sklearn.preprocessing import PolynomialFeatures
x = df["Year"].values.reshape(-1,1)
y = df["Nasa GISS"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x.reshape(-1, 1))
quadratic_model = LinearRegression()
quadratic_model.fit(y=y, X=poly_features)
r2 = quadratic_model.score(y=y, X=poly_features)
print("regression score = ", r2)
print("temperature in 2100 = ", quadratic_model.predict(poly.fit_transform([[2100]])), " C")

##
from sklearn.preprocessing import PolynomialFeatures
x = df["Year"].values.reshape(-1,1)
y = df["Nasa GISS"]
poly10 = PolynomialFeatures(degree=10, include_bias=False)
poly_features = poly10.fit_transform(x.reshape(-1, 1))
ten_model = LinearRegression()
ten_model.fit(y=y, X=poly_features)
r2 = ten_model.score(y=y, X=poly_features)
print("regression score = ", r2)
print("temperature in 2100 = ", ten_model.predict(poly10.fit_transform([[2100]])), " C")

#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

X_modele = np.array(range(1880,2100,1)).reshape(-1,1)

Y_modele = modele.predict(X_modele)

matplotlib.rcParams['font.family'] = 'serif'
plt.scatter(x,y,label = 'données de départ') 
plt.plot(X_modele,Y_modele, '-r',label = 'régression linéraire')
Y_modele = quadratic_model.predict(poly.fit_transform(X_modele))
plt.plot(X_modele,Y_modele, '-b',label = 'régression quadratique')
Y_modele = ten_model.predict(poly10.fit_transform(X_modele))
plt.plot(X_modele,Y_modele, '-g',label = 'régression degré 10')
plt.title("Regression univariée")
plt.xlabel('année')
plt.ylabel('température (C)')
plt.legend(loc='upper left', frameon=False)
plt.savefig('linearregression.pdf')
