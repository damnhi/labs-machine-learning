# %%
from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

# %%
from sklearn.model_selection import train_test_split

X = data_breast_cancer.data
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dt_clf = DecisionTreeClassifier()
log_clf = LogisticRegression()
knn_clf = KNeighborsClassifier()

voting_clf_hard = VotingClassifier(
    estimators=[('dt', dt_clf), ('log', log_clf), ('knn', knn_clf)],
    voting='hard'
)

voting_clf_soft = VotingClassifier(
    estimators=[('dt', dt_clf), ('log', log_clf), ('knn', knn_clf)],
    voting='soft'
)


# %%
from sklearn.metrics import accuracy_score
import pickle

acc_vote = []

for clf in (dt_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft):
    clf.fit(X_train[['mean texture', 'mean symmetry']], y_train)
    y_pred = clf.predict(X_test[['mean texture', 'mean symmetry']])
    acc_vote.append((accuracy_score(y_train, clf.predict(X_train[['mean texture', 'mean symmetry']])), accuracy_score(y_test, y_pred)))
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(acc_vote, f)

with open('vote.pkl', 'wb') as f:
    pickle.dump([dt_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft], f)

# %%
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, random_state=42)
bag_clf_50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, random_state=42)
pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, random_state=42)
pas_clf_50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, max_samples=0.5, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=30, random_state=42)
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=30, random_state=42)
grad_clf = GradientBoostingClassifier(n_estimators=30, random_state=42)

acc_bag = []

for clf in (bag_clf, bag_clf_50, pas_clf, pas_clf_50, rf_clf, ada_clf, grad_clf):
    clf.fit(X_train[['mean texture', 'mean symmetry']], y_train)
    y_pred = clf.predict(X_test[['mean texture', 'mean symmetry']])
    acc_bag.append((accuracy_score(y_train, clf.predict(X_train[['mean texture', 'mean symmetry']])), accuracy_score(y_test, y_pred)))
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(acc_bag, f)

with open('bag.pkl', 'wb') as f:
    pickle.dump([bag_clf, bag_clf_50, pas_clf, pas_clf_50, rf_clf, ada_clf, grad_clf], f)




# %%

bag_clf_fea = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_features=2, max_samples=0.5, bootstrap=True, bootstrap_features=False, random_state=42)

bag_clf_fea.fit(X_train, y_train)
y_pred = bag_clf_fea.predict(X_test)
acc_fea = [(accuracy_score(y_train, bag_clf_fea.predict(X_train)), accuracy_score(y_test, y_pred))]
print(bag_clf_fea.__class__.__name__, accuracy_score(y_test, y_pred))

with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(acc_fea, f)

with open('fea.pkl', 'wb') as f:
    pickle.dump([bag_clf_fea], f)

# %%
import pandas as pd

acc_fea_rank = []

for estimator, features in zip(bag_clf_fea.estimators_, bag_clf_fea.estimators_features_):
    estimator.fit(X_train.iloc[:, features], y_train)
    y_pred = estimator.predict(X_test.iloc[:, features])
    acc_fea_rank.append((accuracy_score(y_train, estimator.predict(X_train.iloc[:, features])), accuracy_score(y_test, y_pred), X_train.columns[features].tolist()))

acc_fea_rank_df = pd.DataFrame(acc_fea_rank, columns=['train_accuracy', 'test_accuracy', 'features'])
acc_fea_rank_df = acc_fea_rank_df.sort_values(by=['train_accuracy', 'test_accuracy'], ascending=False)

with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(acc_fea_rank_df, f)



