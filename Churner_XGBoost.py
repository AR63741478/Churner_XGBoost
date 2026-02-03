from xgboost import XGBRFClassifier as xgb
from xgboost import plot_importance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


df = pd.read_csv("/Users/arhaann/Downloads/Churn.csv")
X = df[["AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage", "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"]]
y = df["Churn"]

sns.countplot(data=df, x='Churn')
plt.title('Churn Imbalace')
plt.show()

model = xgb(objective='binary:logistic', n_estimators=25, eval_metric='auc')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
model.fit(X_train, y_train, verbose=True, eval_set = [(X_test, y_test)])

y_pred = (model.predict_proba(X_test)[:, 1] > 0.35).astype('float')
cm = confusion_matrix(y_test, y_pred, labels = (0, 1))
print(cm)

plot_importance(model)
plt.title("Feature Weights")
plt.show()

print(classification_report(y_test, y_pred))