# Carbon Footprint & Recycling Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Dummy dataset example
data = pd.DataFrame({
    'recycles_regularly':[0,1,0,1,1],
    'carbon_footprint':[120,80,140,70,60]
})

X = data[['recycles_regularly']]
y_reg = data['carbon_footprint']

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2)

model_reg = LinearRegression()
model_reg.fit(X_train, y_train)
pred = model_reg.predict(X_test)

print("RMSE:", mean_squared_error(y_test, pred, squared=False))

# Classification example
y_clf = (data['carbon_footprint'] > 100).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2)

model_clf = LogisticRegression()
model_clf.fit(X_train, y_train)
pred2 = model_clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, pred2))
