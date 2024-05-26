## Machine Learning Lab Summaries

### Lab 01 - Introduction to Data Handling
Initiated with acquiring California housing price data, setting up the work environment, and data analysis. Utilized `matplotlib` for data visualization and conducted correlation analysis and data preparation for training, including splitting into training and testing sets and saving in `.pkl` format.

### Lab 02 - Classification Techniques
Divided data into training and testing sets. Employed classifiers, including Stochastic Gradient Descent (SGD), for detecting the digit 0 from the MNIST dataset. Determined classifier quality using metrics, accuracy, and cross-validation, and recorded classifier accuracy and cross-validation results in `.pkl` files.

### Lab 03 - Regression Analysis
Performed linear regression, KNN for \( k=3 \) and \( k=5 \), and polynomial regression from 2nd to 5th degree. Analyzed the performance of regression functions and compared their behavior with the data distribution. Documented MSE values for training and testing sets and a list of tuples containing regression objects in `.pkl` files.

### Lab 04 - Support Vector Machines and Regression
Conducted hyperparameter tuning, data preparation for classification, and regression using SVM. Utilized breast cancer and iris datasets. Tasks included:
- Classification using SVM with and without feature scaling, accuracy evaluation, and results saved in `bc_acc.pkl` and `iris_acc.pkl`.
- Regression with polynomial feature expansion and LinearSVR, followed by SVR with grid search for hyperparameter optimization. Documented MSE values and saved in `reg_mse.pkl`.


### Lab 05 - Decision Trees
#### 1. Data Preparation
The breast cancer wisconsin dataset and a custom dataset `df` were prepared for decision tree classification and regression tasks.

#### 2. Classification
1. Used decision trees for classifying the `data_breast_cancer` dataset based on the 'mean texture' and 'mean symmetry' features.
2. Split the dataset into 80:20 train-test proportions.
3. Determined the optimal tree depth to maximize F1 score for both train and test sets.
4. Generated a visualization of the decision tree and saved it as `bc.png`.
5. Saved tree depth, F1 scores for train and test sets, and accuracies for train and test sets in a Pickle file `f1acc_tree.pkl`.

#### 3. Regression
1. Used decision trees to build a regressor on the `df` dataset.
2. Split the dataset into 80:20 train-test proportions.
3. Found the optimal tree depth to minimize mean squared error (MSE) for both train and test sets, considering overfitting.
4. Plotted all data points with regressor predictions and compared results with polynomial regression and KNN from previous exercises.
5. Generated a visualization of the decision tree and saved it as `reg.png`.
6. Saved tree depth, MSE for train and test sets in a Pickle file `mse_tree.pkl`.

### Lab 06 - Ensemble Methods
Introduced ensemble methods including parallel and sequential methods, hard/soft voting, bagging, and boosting.

### Lab 07 - Clustering
Explored clustering techniques and parameter tuning for clustering algorithms.

