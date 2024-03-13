# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix




# Training dataset
df = pd.read_csv("training set.csv")

df = df.drop(df.columns[[32]], axis=1) 

# (True or False) There are missing values.
df.isnull().values.any()

# Feature reduction (based off group discussions)
predictors = ["minutes_played", 
              "three_point_percentage", 
              "two_point_percentage", 
              "effective_field_goal_percentage", 
              "free_throw_percentage", 
              "offensive_rebounds", 
              "defensive_rebounds", 
              "assists", 
              "steals", 
              "blocks", 
              "turnovers", 
              "personal_fouls", 
              "points"]

X_train = df[predictors]
y_train = df["all_star"]

# Basic summary statistics of independent variables
X_train.describe(include='all').round(2)

# Histograms of independent variables
for i in X_train:
    plt.figure()
    plt.title(f'{i}')
    plt.hist(X_train[i])

# Boxplot with All-Star versus regular player
for n in X_train:
    ax = sns.boxplot(x=y_train, y=n, data=df)
    plt.show()

# Correlation heatmap of independent variables
corr = X_train.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=3)

# EDA chart based on position
df_pos = df[df["all_star"] == 1]     

cross_tab = pd.crosstab(index=df_pos['season'],
                        columns=df_pos['position'])
cross_tab

cross_tab.plot(kind='bar', stacked=True)
plt.legend()                                                          
plt.xlabel("Season")
plt.show()   




# Test dataset 
df_test = pd.read_csv("test set for prediction.csv")

# Add the actual All-Stars to the dataset (to be used later for accuracy)
actual_2023_all_stars = [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
                         0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,
                         0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
                         0,0,0,0,0,0,0]
df_test["y_actual"] = actual_2023_all_stars

# (True or False) There is missing data.
df_test.isnull().values.any()

X_test = df_test[predictors]
y_actual = df_test["y_actual"]




# Decision Tree #1 - Unpruned/pure leaves
feat_name = list(X_train.columns)
dt = tree.DecisionTreeClassifier(random_state=666)

dt_full = dt.fit(X_train, y_train)

# This provides a text and visual representation (but not pretty)
plot_tree(dt_full, feature_names=feat_name, 
          class_names=["regular player", "All-Star player"])

# Prettier visual representation
plt.figure(figsize=(60, 20), dpi=400)
tree.plot_tree(dt_full, 
               fontsize=4,
               feature_names=feat_name, 
               class_names=["regular player", "All-Star player"],
               filled=True)
plt.show()

# Prettier text representation
dt_full_text = export_text(dt_full, feature_names=feat_name)
print(dt_full_text)

# Decision tree predictions/accuracy/confusion matrix
dt_y_hat = dt_full.predict(X_test) 

accuracy_score(y_actual, dt_y_hat)

dt_conf = confusion_matrix(y_actual, dt_y_hat)




# Decision Tree #2 - Stop growing at a depth 
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import make_scorer

# Depth of dt_full is 13
max_depth = dt_full.get_depth()
max_depth

# Find optimal depth, which is 4.
max_depth_grid = GridSearchCV(
    estimator=dt,
    scoring=make_scorer(accuracy_score),
    param_grid={'max_depth': np.arange(1, max_depth + 1, 1)}
    )

max_depth_grid.fit(X_train, y_train)

max_depth_grid.best_params_

dt_depth4 = max_depth_grid.best_estimator_

# Plot tree with max depth of 4
plt.figure(figsize=(15, 7), dpi=200)
plot_tree(
    dt_depth4,
    feature_names=feat_name,
    fontsize=4,
    class_names=["regular player", "All-Star player"],
    filled=True
)
plt.show()

# Text representation
dt_depth4_text = export_text(dt_depth4, feature_names=feat_name)
print(dt_depth4_text)

# Predictions/accuracy
dt_depth4_y_hat = dt_depth4.predict(X_test) 

accuracy_score(y_actual, dt_depth4_y_hat)




# Decision Tree #3 & #4 - Prune with alpha
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas

dts = []
for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=666, ccp_alpha=ccp_alpha)
    dt.fit(X_train, y_train)
    dts.append(dt)
print(
    "Number of nodes in the last tree is {} with ccp_alpha: {}".format(
    dts[-1].tree_.node_count, ccp_alphas[-1]
    )
)

dts = dts[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [dt.tree_.node_count for dt in dts]
depth = [dt.tree_.max_depth for dt in dts]
fig, ax = plt.subplots(2, figsize=(10,10))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("depth vs alpha")
fig.tight_layout()

train_scores = [dt.score(X_train, y_train) for dt in dts]
test_scores = [dt.score(X_test, y_actual) for dt in dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Alpha for Training and Testing Datasets")
ax.plot(ccp_alphas, 
        train_scores, 
        marker="o", 
        label="train", 
        drawstyle="steps-post")
ax.plot(ccp_alphas, 
        test_scores, 
        marker="o", 
        label="test", 
        drawstyle="steps-post")
ax.legend()
plt.show()

# dt with ccp_alpha at 0.0010
dt_alpha_10 = tree.DecisionTreeClassifier(random_state=666, 
    ccp_alpha=0.0010)
dt_alpha_10.fit(X_train, y_train)

# Visual representation of ccp_alpha 0.0010 decision tree
plt.figure(figsize=(10, 5), dpi=200)
tree.plot_tree(dt_alpha_10, 
               fontsize=4, 
               feature_names=feat_name, 
               class_names=["regular player", "All-Star player"],
               filled=True)
plt.suptitle('ccp_alpha 0.0010', fontsize=12)
plt.show()

# Text representation of ccp_alpha 0.0010 decision tree
dt_alpha_10_text = export_text(dt_alpha_10, feature_names=feat_name)
print(dt_alpha_10_text)

# Prediction/accuracy
dt_a10_y_hat = dt_alpha_10.predict(X_test)

accuracy_score(y_actual, dt_a10_y_hat)


# dt with ccp_alpha at 0.0012
dt_alpha_12 = tree.DecisionTreeClassifier(random_state=666, 
    ccp_alpha=0.0012)
dt_alpha_12.fit(X_train, y_train)

# Visual representation of ccp_alpha 0.0012 decision tree
plt.figure(figsize=(10, 5), dpi=200)
tree.plot_tree(dt_alpha_12, 
               fontsize=5, 
               feature_names=feat_name, 
               class_names=["regular player", "All-Star player"],
               filled=True)
plt.suptitle('ccp_alpha 0.0012', fontsize=12)
plt.show()

# Text representation of ccp_alpha 0.0012 decision tree
dt_alpha_12_text = export_text(dt_alpha_12, feature_names=feat_name)
print(dt_alpha_12_text)

# Prediction/accuracy
dt_a12_y_hat = dt_alpha_12.predict(X_test)

accuracy_score(y_actual, dt_a12_y_hat)




# Random Forest - 100 trees
rf = RandomForestClassifier(n_estimators = 100, 
                            max_features = "sqrt",
                            random_state=42)
rf = rf.fit(X_train, y_train)

# Feature importance
rf_100_feat_imp_sorted = pd.Series(
    rf.feature_importances_, index=feat_name
    ).sort_values(ascending=False)
rf_100_feat_imp_sorted

# Prediction/accuracy/confusion matrix
rf_y_hat = rf.predict(X_test) 

accuracy_score(y_actual, rf_y_hat)

rf_conf = confusion_matrix(y_actual, rf_y_hat)
rf_conf




# Random Forest - 500 trees
rf_500 = RandomForestClassifier(n_estimators=500, 
                            max_features = "sqrt",
                            random_state=42)
rf_500 = rf_500.fit(X_train, y_train)

# Feature importance
rf_500_feat_imp_sorted = pd.Series(
    rf_500.feature_importances_, 
    index=feat_name).sort_values(ascending=False)
rf_500_feat_imp_sorted

# Prediction/accuracy
rf_500_y_hat = rf_500.predict(X_test) 

accuracy_score(y_actual, rf_500_y_hat)




# Random Forest - 1000 trees
rf_1000 = RandomForestClassifier(n_estimators=1000, 
                            max_features = "sqrt",
                            random_state=42)
rf_1000 = rf_1000.fit(X_train, y_train)

# Feature importance
rf_1000_feat_imp_sorted = pd.Series(
    rf_1000.feature_importances_, 
    index=feat_name).sort_values(ascending=False)
rf_1000_feat_imp_sorted

# Prediction/accuracy
rf_1000_y_hat = rf_1000.predict(X_test) 

accuracy_score(y_actual, rf_1000_y_hat)




# Plot random forests' feature importances
rf_imp = pd.DataFrame({"100 trees": rf_100_feat_imp_sorted})
rf_imp["500 trees"] = rf_500_feat_imp_sorted
rf_imp["1000 trees"] = rf_1000_feat_imp_sorted
rf_imp

ax = rf_imp.plot.barh(figsize=(10,10), 
                      width=0.85, 
                      fontsize=20,
                      color={"100 trees": "#343032",
                             "500 trees": "#82787D",
                             "1000 trees": "#C7C1C4"})
ax.legend(fontsize=20)




# Since the random forest with 500 trees (the model with the highest 
# accuracy) only predicted 21 All-Star players, let's look at the 
# next couple of players whose majority vote was closest to being 
# an "All-Star" (class 1).
Names = df_test["player"]
prob_500 = rf_500.predict_proba(X_test)
prob_500 = pd.DataFrame(prob_500, index=Names)
prob_500

almost_all_star = prob_500[(prob_500[1]>=0.42) & (prob_500[1]<0.5)]
almost_all_star

