# -*- coding: utf-8 -*-
"""
Created on Sun May 25 22:27:30 2025

@author: maoer
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from imblearn.over_sampling import SMOTE

import matplotlib
print(matplotlib.__version__)



fig, ax = plt.subplots()
bars = ax.bar(['A', 'B', 'C'], [5, 10, 7])
ax.bar_label(bars)
plt.show()



df = pd.read_csv(r"D:\chrunmodeling\Churn_Modelling.csv")

df.head()

 # Checking the Dimensions of Dataset.
print("Total number of records/rows present in the dataset is:",df.shape[0])
print("Total number of attributes/columns present in the dataset is:",df.shape[1])

df.columns
df.info()

df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})

df[df.duplicated()]

df.describe(include="object").T

df.sample(5)

# Dropping Insignificant Features.

df.drop(columns=["RowNumber","CustomerId","Surname"],inplace=True)

# 3. Renaming Target Variable name and its values with more appropirate values for better Analysis.

df.rename(columns={"Exited":"Churned"},inplace=True)
df["Churned"].replace({0:"No",1:"Yes"},inplace=True)
df.head()

# Explorator Data Analysis
count = df["Churned"].value_counts()

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
ax=sns.countplot(df["Churned"],palette="Set2")
ax.bar_label(ax.containers[0],fontweight="black",size=15)
plt.title("Customer Churned Disribution",fontweight="black",size=20,pad=20)

plt.subplot(1,2,2)
plt.pie(count.values, labels=count.index, autopct="%1.1f%%",colors=sns.set_palette("Set2"),
        textprops={"fontweight":"black"},explode=[0,0.1])
plt.title("Customer Churned Disribution",fontweight="black",size=20,pad=20)
plt.show()

# Visualizing Customer Churned by Gender.
def countplot(column):
    plt.figure(figsize=(15,5))
    ax = sns.countplot(x=column, data=df, hue="Churned",palette="Set2")
    for value in ax.patches:
        percentage = "{:.1f}%".format(100*value.get_height()/len(df[column]))
        x = value.get_x() + value.get_width() / 2 - 0.05
        y = value.get_y() + value.get_height()
        ax.annotate(percentage, (x,y), fontweight="black",size=15)
        
    plt.title(f"Customer Churned by {column}",fontweight="black",size=20,pad=20)
    plt.show()
    
countplot("Gender")

# Visualizing Customer Churned by Geoprahical Region.
countplot("Geography")

# Visualizing Customer Churn by "HasCrCard".
countplot("HasCrCard")

# Visualizing Customer Churned by "NumOfProducts".
countplot("NumOfProducts")

# Visualizing Customer Churned by "IsActiveMember".
countplot("IsActiveMember")

# Visualizing Customer Churned by "Tenure".
plt.figure(figsize=(15,5))
ax = sns.countplot(x="Tenure", data=df, hue="Churned",palette="Set2")
for value in ax.patches:
    percentage = "{:.1f}%".format(100*value.get_height()/len(df["Tenure"]))
    x = value.get_x() + value.get_width() / 2 - 0.05
    y = value.get_y() + value.get_height()
    ax.annotate(percentage, (x,y), fontweight="black",size=12, ha="center")

plt.title("Customer Churned by Tenure",fontweight="black",size=20,pad=20)
plt.show()

# Visualizing Customer Churned by "CreditScore".
def continous_plot(column):
    plt.figure(figsize=(10, 5))
    
    # Boxplot (updated syntax)
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="Churned", y=column, palette="Set2")  # ✅ Fixed
    plt.title(f"Boxplot of {column} vs Churned")
    
    # Distribution plot
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column, hue="Churned", kde=True, palette="Set2")
    plt.title(f"Distribution of {column}")
    
    plt.tight_layout()
    plt.show()

continous_plot("CreditScore")

 # 9Visualizing Customer Churned by "Age".
continous_plot("Age")

# 10.Visualizing Customer Churned by "Balance."
continous_plot("Balance")


# 11.Visualizing Customer Churned by "Estimated Salary".
 
continous_plot("EstimatedSalary") 



# Feature Engineering
# 1. Creating New Feature From "NumOfProducts" Feature.
conditions = [(df["NumOfProducts"]==1), (df["NumOfProducts"]==2), (df["NumOfProducts"]>2)]
values =     ["One product","Two Products","More Than 2 Products"]

df["Total_Products"] = np.select(conditions,values)
df.drop(columns="NumOfProducts", inplace=True)

# Visualizing The New Feature "Total_Products".
countplot("Total_Products")

# 2. Creating New Feature From "Balance" Feature.
conditions = [(df["Balance"]==0), (df["Balance"]>0)]
values = ["Zero Balance","More Than zero Balance"]
df["Account_Balance"] = np.select(conditions, values)

df.drop(columns="Balance",inplace=True)

# Visualizing The New Feature "Account_Balance".
countplot("Account_Balance")

# Data Preprocessing
# 1. Computing Unique Values of Categorical Columns.
cat_cols = ["Geography","Gender","Total_Products","Account_Balance"]

for column in cat_cols:
    print(f"Unique Values in {column} column is:",df[column].unique())
    print("-"*100,"\n")
    
# 2. Performing One Hot Encoding on Categorical Features.   
df = pd.get_dummies(columns=cat_cols, data=df)

# 3. Encoding Target Variable.
df["Churned"].replace({"No":0,"Yes":1},inplace=True)
df.head()

# 4. Checking Skewness of Continous Features.
cols = ["CreditScore","Age","EstimatedSalary"]
df[cols].skew().to_frame().rename(columns={0:"Feature Skewness"})

# 4. Performing Log Transformation on Age Column.
old_age = df["Age"]     ##Storing the previous Age values to compare these values with the transformed values.

df["Age"] = np.log(df["Age"])

# 5. Visualizing Age Before and After Transformation.

plt.figure(figsize=(13,6))
plt.subplot(1,2,1)
sns.histplot(old_age, color="purple", kde=True)
plt.title("Age Distribution Before Transformation",fontweight="black",size=18,pad=20)

plt.subplot(1,2,2)
sns.histplot(df["Age"], color="purple", kde=True)
plt.title("Age Distribution After Transformation",fontweight="black",size=18,pad=20)
plt.tight_layout()
plt.show()

# 6. Segregating Features & Labels for Model Training.
X = df.drop(columns=["Churned"])
y = df["Churned"]

print(type(df))
print(df.head())
print(df.columns)

# 7. Splitting Data For Model Training & Testing.


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Shape of x_train is:",x_train.shape)
print("Shape of x_test is: ",x_test.shape)
print("Shape of y_train is:",y_train.shape)
print("Shape of y_test is: ",y_test.shape)



# 8. Applying SMOTE to Overcome the Class-Imbalance in Target Variable.
smt = SMOTE(random_state=42)
x_train_resampled,y_train_resampled = smt.fit_resample(x_train,y_train)
print(x_train_resampled.shape ,y_train_resampled.shape)

y_train_resampled.value_counts().to_frame()


# Model Creation using DecisionTree
# 1. Performing Grid-Search with cross-validation to find the best Parameters for the Model.
dtree = DecisionTreeClassifier()
param_grid = {"max_depth":[3,4,5,6,7,8,9,10],
              "min_samples_split":[2,3,4,5,6,7,8],
              "min_samples_leaf":[1,2,3,4,5,6,7,8],
              "criterion":["gini","entropy"],
              "splitter":["best","random"],
              "max_features":["auto",None],
              "random_state":[0,42]}
grid_search = GridSearchCV(dtree, param_grid, cv=5, n_jobs=-1)

grid_search.fit(x_train_resampled,y_train_resampled)

# 2. Fetching the Best Parameters for DecisionTree Model.
best_parameters = grid_search.best_params_

print("Best Parameters for DecisionTree Model is:\n\n")
best_parameters



# 3. Creating DecisionTree Model Using Best Parameters.
dtree = DecisionTreeClassifier(**best_parameters)

dtree.fit(x_train_resampled,y_train_resampled)


# 4. Computing Model Accuracy.
y_train_pred = dtree.predict(x_train_resampled)
y_test_pred = dtree.predict(x_test)

print("Accuracy Score of Model on Training Data is =>",round(accuracy_score(y_train_resampled,y_train_pred)*100,2),"%")
print("Accuracy Score of Model on Testing Data  is =>",round(accuracy_score(y_test,y_test_pred)*100,2),"%")

# 5. Model Evaluation using Different Metric Values.
print("F1 Score of the Model is =>",f1_score(y_test,y_test_pred,average="micro"))
print("Recall Score of the Model is =>",recall_score(y_test,y_test_pred,average="micro"))
print("Precision Score of the Model is =>",precision_score(y_test,y_test_pred,average="micro"))

# We can observe that recall, precision, and F1 score are all the same, it means that our model is 
# achieving perfect balance between correctly identifying positive samples (recall) and minimizing false positives (precision).
# The high values for F1 score, recall score, and precision score, all of which are approximately 0.8. These metrics suggest
#  that the model achieves good accuracy in predicting the positive class.

# 6. Finding Importance of Features in DecisionTreeClassifier.
imp_df = pd.DataFrame({"Feature Name":x_train.columns,
                       "Importance":dtree.feature_importances_})
features = imp_df.sort_values(by="Importance",ascending=False)

plt.figure(figsize=(12,7))
sns.barplot(x="Importance", y="Feature Name", data=features, palette="plasma")
plt.title("Feature Importance in the Model Prediction", fontweight="black", size=20, pad=20)
plt.yticks(size=12)
plt.show()

# The key factors that significantly influence the deactivation of customers banking facilities are:-
# Total_Products, Age, IsActiveMember, Geography, Balance and Gender.
# The minimal impact of features on the deactivation of customers' banking facilities are:-
# CreditScore, HasCrCard, Tenure and EstimatedSalary

# 7. SHAP Summary Plot: Explaining Model Predictions with Feature Importance.
import shap
import numba
explainer = shap.TreeExplainer(dtree)
shap_values = explainer.shap_values(x_test)

plt.title("Feature Importance and Effects on Predictions",fontweight="black",pad=20,size=18)
shap.summary_plot(shap_values[1], x_test.values, feature_names = x_test.columns,plot_size=(14,8))



# 8. Model Evaluation using Confusion Matrix.
cm = confusion_matrix(y_test,y_test_pred)

plt.figure(figsize=(15,6))
sns.heatmap(data=cm, linewidth=.5, annot=True, fmt="g", cmap="Set1")
plt.title("Model Evaluation using Confusion Matrix",fontsize=20,pad=20,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

# * **Strong True Positive Rate:** The model achieved a high number of true positive predictions, 
# indicating its ability to correctly identify positive cases. This suggests that the model is 
# effective in accurately classifying the desired outcome. * **Need of Improvement in False Negative Rate:** 
# The presence of a relatively high number of false negatives suggests that the model may have missed 
# identifying some actual positive cases. This indicates a need for further refinement to enhance the model's
# ability to capture all positive cases.

# 9. Model Evaluation: ROC Curve and Area Under the Curve (AUC)
y_pred_proba = dtree.predict_proba(x_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=["y_actual"])])
df_actual_predicted.index = y_test.index


fpr, tpr, thresholds = roc_curve(df_actual_predicted["y_actual"], y_pred_proba)
auc = roc_auc_score(df_actual_predicted["y_actual"], y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}",color="green")
plt.plot([0, 1], [0, 1], linestyle="--", color="black")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve",pad=20,fontweight="black")
plt.legend()
plt.show()

# An AUC (Area Under the Curve) value of 0.84 suggests that the model has strong discriminative power.
# This suggests that the model has a high ability to distinguish between positive and negative instances, indicating its effectiveness in making accurate predictions.
# The model has a relatively high probability of ranking a randomly selected positive instance higher than a randomly selected negative instance.

# Model Creation using RandomForest.
# 1. Performing Grid-Search with cross-validation to find the best Parameters for the Model.
rfc = RandomForestClassifier()
param_grid = {"max_depth":[3,4,5,6,7,8],
              "min_samples_split":[3,4,5,6,7,8],
              "min_samples_leaf":[3,4,5,6,7,8],
              "n_estimators": [50,70,90,100],
              "criterion":["gini","entropy"]}

grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)

grid_search.fit(x_train_resampled,y_train_resampled)

# 2. Fetching the Best Parameters for RandomForest Model.
best_parameters = grid_search.best_params_

print("Best Parameters for RandomForest Model is:\n\n")
best_parameters

# 3. Creating RandomForest Model Using Best Parameters.
rfc = RandomForestClassifier(**best_parameters)

rfc.fit(x_train_resampled,y_train_resampled)

# 4. Computing Model Accuracy.
y_train_pred = rfc.predict(x_train_resampled)
y_test_pred  = rfc.predict(x_test)

print("Accuracy Score of Model on Training Data is =>",round(accuracy_score(y_train_resampled,y_train_pred)*100,2),"%")
print("Accuracy Score of Model on Testing Data  is =>",round(accuracy_score(y_test,y_test_pred)*100,2),"%")

# 5. Model Evaluation using Different Metric Values.
print("F1 Score of the Model is =>",f1_score(y_test,y_test_pred,average="micro"))
print("Recall Score of the Model is =>",recall_score(y_test,y_test_pred,average="micro"))
print("Precision Score of the Model is =>",precision_score(y_test,y_test_pred,average="micro"))

# We can observe that recall, precision, and F1 score are all the same, it means that our model is achieving 
# perfect balance between correctly identifying positive samples (recall) and minimizing false positives (precision).
# # The high values for F1 score, recall score, and precision score, all of which are approximately 0.8.
#  These metrics suggest that the model achieves good accuracy in predicting the positive class.

# Finding Importance of Features in RandomForest Model.

imp_df = pd.DataFrame({"Feature Name":x_train.columns,
                       "Importance":rfc.feature_importances_})

features = imp_df.sort_values(by="Importance",ascending=False)

plt.figure(figsize=(12,7))
sns.barplot(x="Importance", y="Feature Name", data=features, palette="plasma")
plt.title("Feature Importance in the Model Prediction", fontweight="black", size=20, pad=20)
plt.yticks(size=12)
plt.show()

# The key factors that significantly influence the deactivation of customers banking facilities are:-
# Total_Products, Age, IsActiveMember, Geography, Gende and Balance.
# The minimal impact of features on the deactivation of customers' banking facilities are:-
# HasCrCard, Tenure, CreditScore and EstimatedSalary

# 7. SHAP Summary Plot: Explaining Model Predictions with Feature Importance.
import shap
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(x_test)

plt.title("Feature Importance and Effects on Predictions",fontweight="black",pad=20,size=18)
shap.summary_plot(shap_values[1], x_test.values, feature_names = x_test.columns,plot_size=(14,8))

# The red color represents high feature values, indicating that the feature positively contributes for increasing the prediction value.
# The blue color represents low feature values, indicating that the feature negatively contributes for decreasing the prediction value.

# 8. Model Evaluation using Confusion Matrix.
cm = confusion_matrix(y_test,y_test_pred)

plt.figure(figsize=(15,6))
sns.heatmap(data=cm, linewidth=.5, annot=True, fmt="g", cmap="Set1")
plt.title("Model Evaluation using Confusion Matrix",fontsize=20,pad=20,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

 # **Strong True Positive Rate:** The model achieved a high number of true positive predictions, 
 # indicating its ability to correctly identify positive cases. This suggests that the model is 
 # effective in accurately classifying the desired outcome. * **Need of Improvement in False Negative Rate:** 
 # The presence of a relatively high number of false negatives suggests that the model may 
 # have missed identifying some actual positive cases. This indicates a need for further 
 # refinement to enhance the model's ability to capture all positive cases.
 
 # 9.Model Evaluation: ROC Curve and Area Under the Curve (AUC)
y_pred_proba = rfc.predict_proba(x_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=["y_actual"])])
df_actual_predicted.index = y_test.index


fpr, tpr, thresholds = roc_curve(df_actual_predicted["y_actual"], y_pred_proba)
auc = roc_auc_score(df_actual_predicted["y_actual"], y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}",color="green")
plt.plot([0, 1], [0, 1], linestyle="--", color="black")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve",pad=20,fontweight="black")
plt.legend()
plt.show()
 
# An AUC (Area Under the Curve) value of 0.86 suggests that the model has strong discriminative power.
# This suggests that the model has a high ability to distinguish between positive and negative instances, indicating its effectiveness in making accurate predictions.
# The model has a relatively high probability of ranking a randomly selected positive instance higher than a randomly selected negative instance.


# Key-Points
# The key factors that significantly influence the deactivation of customers banking facilities are Total_Products, Age, IsActiveMember, 
# Gender and Geography.
# High Training and Testing Accuracies: Both the model achieved a high accuracy score near to 90% on the training data, indicating a 
# good fit to the training instances. Additionally, the model's accuracy score near to 85% on the testing data suggests its ability 
# to generalize well to unseen instances.
# High F1 Score, Recall, and Precision: The model achieved high F1 score, recall, and precision values, all approximately 0.8. This 
# indicates that the model has a strong ability to correctly identify positive cases while minimizing false positives and maximizing 
# true positives.
# High AUC value more than 0.8, states that the model demonstrates a reasonably good discriminatory power. It suggests that the model
#  is able to distinguish between positive and negative instances with a relatively high degree of accuracy.
# Overall Model Performance: The model demonstrates strong performance across multiple evaluation metrics, indicating its effectiveness
#  in making accurate predictions and capturing the desired outcomes.


#  Recommendations
# The bank can try to convince the customers to have atleast 2 banking products but not less than 2.
# The bank can launch a scheme for customers with higher ages (Senior Citizens) so that they not deactivate their banking facilities.
# The bank can provide Rewards and Incentive Programs, Regular Communication and Updates, and Enhanced Digital Services 
# so that customers remain active to the banking facilities.

# XGboost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 1. Initialize model and set parameter grid
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 2. Grid Search tuning
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search.fit(x_train_resampled, y_train_resampled)

# 3. Get optimal parameters
best_params = grid_search.best_params_
print("Best Parameters for XGBoost:\n", best_params)

# 4. Model Training
xgb_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train_resampled, y_train_resampled)

# 5. Model Evaluation
y_train_pred = xgb_model.predict(x_train_resampled)
y_test_pred = xgb_model.predict(x_test)

print("\nTraining Accuracy:", round(accuracy_score(y_train_resampled, y_train_pred)*100, 2), "%")
print("Testing Accuracy:", round(accuracy_score(y_test, y_test_pred)*100, 2), "%")
print("F1 Score:", f1_score(y_test, y_test_pred, average="macro"))
print("AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(x_test)[:, 1]))

# 6. Feature Importance analysis
plt.figure(figsize=(12, 7))
xgb.plot_importance(xgb_model, height=0.8)
plt.title("XGBoost Feature Importance", fontweight="black", size=20, pad=20)
plt.show()

# 7. SHAP Explanation
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test, plot_size=(14, 8))

cm = confusion_matrix(y_test,y_test_pred)

plt.figure(figsize=(15,6))
sns.heatmap(data=cm, linewidth=.5, annot=True, fmt="g", cmap="Set1")
plt.title("Model Evaluation using Confusion Matrix",fontsize=20,pad=20,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Obtain predicted probabilities (using XGBoost as an example, the same applies to LightGBM)
y_prob = xgb_model.predict_proba(x_test)[:, 1]  # The probability of positive

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#20B2AA', lw=3, 
         label=f'XGBoost (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='#4169E1', 
         label='Random Guess (AUC = 0.500)')

# Set Style
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
plt.title('ROC Curve - XGBoost', fontsize=16, pad=20, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()



#lighGBM
import lightgbm as lgb

# 1. Initialize model and set parameter grid
lgb_model = lgb.LGBMClassifier()
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100, 200],
    'num_leaves': [31, 50, 100],
    'min_child_samples': [20, 50],
    'subsample': [0.8, 1.0]
}

# 2. Grid Search tuning
grid_search = GridSearchCV(lgb_model, param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search.fit(x_train_resampled, y_train_resampled)

# 3. Get optimal parameters
best_params = grid_search.best_params_
print("Best Parameters for LightGBM:\n", best_params)

# 4. Model training
lgb_model = lgb.LGBMClassifier(**best_params)
lgb_model.fit(x_train_resampled, y_train_resampled)

# 5. Model Evaluation
y_train_pred = lgb_model.predict(x_train_resampled)
y_test_pred = lgb_model.predict(x_test)

print("\nTraining Accuracy:", round(accuracy_score(y_train_resampled, y_train_pred)*100, 2), "%")
print("Testing Accuracy:", round(accuracy_score(y_test, y_test_pred)*100, 2), "%")
print("F1 Score:", f1_score(y_test, y_test_pred, average="macro"))
print("AUC Score:", roc_auc_score(y_test, lgb_model.predict_proba(x_test)[:, 1]))

# 6. Feature Importance analysis
lgb.plot_importance(lgb_model, height=0.8, figsize=(12, 7))
plt.title("LightGBM Feature Importance", fontweight="black", size=20, pad=20)
plt.show()

# 7. SHAP Explanation
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test, plot_size=(14, 8))


cm = confusion_matrix(y_test,y_test_pred)

plt.figure(figsize=(15,6))
sns.heatmap(data=cm, linewidth=.5, annot=True, fmt="g", cmap="Set1")
plt.title("Model Evaluation using Confusion Matrix",fontsize=20,pad=20,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Obtain predicted probabilities (using XGBoost as an example, the same applies to LightGBM)
y_prob = lgb_model.predict_proba(x_test)[:, 1]  # The probability of positive

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='#20B2AA', lw=3, 
         label=f'lightGBM (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='#4169E1', 
         label='Random Guess (AUC = 0.500)')

# Set Style
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
plt.title('ROC Curve - lightGBM', fontsize=16, pad=20, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

##Random forest provides the best model.

# The predicted probabilities from multiple models
xgb_probs = xgb_model.predict_proba(x_test)[:, 1]
lgb_probs = lgb_model.predict_proba(x_test)[:, 1]
dtree_probs= dtree.predict_proba(x_test)[:][:,1]
rf_probs = rfc.predict_proba(x_test)[:, 1]  # random forest model

plt.figure(figsize=(10, 6))

# Plot ROC curve from multiple models
models = {
    'XGBoost': xgb_probs,
    'LightGBM': lgb_probs,
    'Random Forest': rf_probs,
    'DecisionTree': dtree_probs 
}

colors = ['#FF6347', '#20B2AA', '#9370DB']

for (name, probs), color in zip(models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, lw=2.5, color=color,
             label=f'{name} (AUC = {auc_score:.3f})')

# Set Style
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
plt.title('Model Comparison - ROC Curves', fontsize=16, pad=20, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
