# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split


# %%
data_breast_cancer = datasets.load_breast_cancer(as_frame=True) 
print(data_breast_cancer['DESCR'])

# %%
data_iris = datasets.load_iris(as_frame=True)
print(data_iris['DESCR'])

# %%
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


X_bc = data_breast_cancer['data'][['mean area', 'mean smoothness']]
y_bc = data_breast_cancer['target']

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

model_without_auto_scaling = LinearSVC(loss='hinge' ,random_state=42)
model_without_auto_scaling.fit(X_train_bc, y_train_bc)
y_train_bc_pred = model_without_auto_scaling.predict(X_train_bc)
y_test_bc_pred = model_without_auto_scaling.predict(X_test_bc)

model_with_auto_scaling = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(loss='hinge', random_state=42))
])
model_with_auto_scaling.fit(X_train_bc, y_train_bc)
y_train_bc_pred_scaled = model_with_auto_scaling.predict(X_train_bc)
y_test_bc_pred_scaled = model_with_auto_scaling.predict(X_test_bc)

accuracy = [
    accuracy_score(y_train_bc, y_train_bc_pred),
    accuracy_score(y_test_bc, y_test_bc_pred),
    accuracy_score(y_train_bc, y_train_bc_pred_scaled),
    accuracy_score(y_test_bc, y_test_bc_pred_scaled)
]

with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(accuracy, f)

# %%
X_ir = data_iris['data'][['petal length (cm)', 'petal width (cm)']]
y_ir = (data_iris['target'] == 2).astype(np.float64)
X_train_ir, X_test_ir, y_train_ir, y_test_ir = train_test_split(X_ir, y_ir, test_size=0.2, random_state=42)

model_iris_without_auto_scaling = LinearSVC(loss='hinge', random_state=42)
model_iris_without_auto_scaling.fit(X_train_ir, y_train_ir)
y_train_ir_pred = model_iris_without_auto_scaling.predict(X_train_ir)
y_test_ir_pred = model_iris_without_auto_scaling.predict(X_test_ir)

model_iris_with_auto_scaling = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(loss='hinge', random_state=42))])
model_iris_with_auto_scaling.fit(X_train_ir, y_train_ir)
y_train_ir_pred_scaled = model_iris_with_auto_scaling.predict(X_train_ir)
y_test_ir_pred_scaled = model_iris_with_auto_scaling.predict(X_test_ir)

accuracy_iris = [
    accuracy_score(y_train_ir, y_train_ir_pred),
    accuracy_score(y_test_ir, y_test_ir_pred),
    accuracy_score(y_train_ir, y_train_ir_pred_scaled),
    accuracy_score(y_test_ir, y_test_ir_pred_scaled)
]

with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(accuracy_iris, f)



# %%
size = 900
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR , SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

poly_features = PolynomialFeatures(degree=4)
X_train_reg_poly = poly_features.fit_transform(X_train_reg.reshape(-1, 1))
X_test_reg_poly = poly_features.transform(X_test_reg.reshape(-1, 1))

svm_reg_poly = LinearSVR()
svm_reg_poly.fit(X_train_reg_poly, y_train_reg)
y_train_reg_poly_pred = svm_reg_poly.predict(X_train_reg_poly)
y_test_reg_poly_pred = svm_reg_poly.predict(X_test_reg_poly)

mse_train_reg_poly = mean_squared_error(y_train_reg, y_train_reg_poly_pred)
mse_test_reg_poly = mean_squared_error(y_test_reg, y_test_reg_poly_pred)


# %%
svm_poly_reg = SVR(kernel='poly', degree=4)
svm_poly_reg.fit(X_train_reg.reshape(-1, 1), y_train_reg)
y_train_reg_pred = svm_poly_reg.predict(X_train_reg.reshape(-1, 1))
y_test_reg_pred = svm_poly_reg.predict(X_test_reg.reshape(-1, 1))

mse_train_reg = mean_squared_error(y_train_reg, y_train_reg_pred)
mse_test_reg = mean_squared_error(y_test_reg, y_test_reg_pred)


# %%
param_grid = {
    'C': [0.1, 1, 10],
    'coef0': [0.1, 1, 10]
}

search = GridSearchCV(svm_poly_reg, param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
search.fit(X.reshape(-1, 1), y)

svm_poly_reg_best = SVR(kernel='poly', degree=4, C=10, coef0=10)
svm_poly_reg_best.fit(X_train_reg.reshape(-1, 1), y_train_reg)
y_train_reg_pred_best = svm_poly_reg_best.predict(X_train_reg.reshape(-1, 1))
y_test_reg_pred_best = svm_poly_reg_best.predict(X_test_reg.reshape(-1, 1))
mse_train_reg_best = mean_squared_error(y_train_reg, y_train_reg_pred_best)
mse_test_reg_best = mean_squared_error(y_test_reg, y_test_reg_pred_best)

# %%
mse_result_list = [mse_train_reg_poly, mse_test_reg_poly, mse_train_reg_best, mse_test_reg_best]
with open('reg_mse.pkl', 'wb') as f:
    pickle.dump(mse_result_list, f)

# %%



