import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data=pd.read_csv('Banknote_authentication.csv')
data.head()
sns.heatmap(data.corr(),annot=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')   # no null values
sns.set_style('whitegrid')
sns.countplot(x='class',data=data,palette='coolwarm')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('class',axis=1), 
                                                    data['class'], test_size=0.20, random_state=0)
                                                    
X_train
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel.score(X_train,y_train)   #0.9899
logmodel.score(X_test,y_test)     #0.9927
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
roc_auc=metrics.auc(fpr,tpr)
roc_auc

plt.plot(fpr,tpr)
plt.legend()
plt.show()
