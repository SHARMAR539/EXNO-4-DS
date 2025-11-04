# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE



```
import pandas as pd
import numpy as np
import seaborn as sns
```

```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

```
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```

<img width="1488" height="425" alt="image" src="https://github.com/user-attachments/assets/e4b3d993-c62b-4fae-8426-2bcae7eed4b5" />


```
data.isnull().sum()
```

<img width="227" height="495" alt="image" src="https://github.com/user-attachments/assets/4cbccaa8-e27d-4eee-946d-a72d16eaff80" />


```
missing=data[data.isnull().any(axis=1)]
missing
```


<img width="1402" height="422" alt="image" src="https://github.com/user-attachments/assets/5d319fa0-6b93-420f-a12c-124754c85e2b" />


```
data2=data.dropna(axis=0)
data2
```


<img width="1421" height="423" alt="image" src="https://github.com/user-attachments/assets/5f700bfe-eb45-4fd1-aca6-a54dc612dedb" />


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```


<img width="1156" height="334" alt="image" src="https://github.com/user-attachments/assets/bd802c23-7d2c-4a09-9b8d-eb3a6af03f72" />


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="391" height="425" alt="image" src="https://github.com/user-attachments/assets/c04ae58b-4304-42c6-991d-6516b26bbfca" />


```
data2
```


<img width="1323" height="420" alt="image" src="https://github.com/user-attachments/assets/690bb39a-2b05-46b5-839c-5e8d6b90ad45" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```


<img width="1821" height="515" alt="image" src="https://github.com/user-attachments/assets/28074023-c8b1-4728-9ca6-a9ae4c1cfead" />


```
columns_list=list(new_data.columns)
print(columns_list)
```


<img width="1783" height="63" alt="image" src="https://github.com/user-attachments/assets/036beac1-9605-427e-9098-9492303ca546" />


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```


<img width="1802" height="52" alt="image" src="https://github.com/user-attachments/assets/3bbec95c-0b82-4af0-a13e-5d826b97f3e4" />


```
y=new_data['SalStat'].values
print(y)
```

<img width="222" height="32" alt="image" src="https://github.com/user-attachments/assets/f5da65b1-3b92-4f41-8541-fcbef549faa9" />



```
x=new_data[features].values
print(x)
```


<img width="449" height="142" alt="image" src="https://github.com/user-attachments/assets/9ad56aab-f2f8-494e-896c-2db74dcb5f91" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size = 0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```


<img width="329" height="86" alt="image" src="https://github.com/user-attachments/assets/76cf17d7-fa3a-4f34-9018-46b9573f835c" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="208" height="58" alt="image" src="https://github.com/user-attachments/assets/07597e97-117f-4a32-a83d-a2438c34b1f2" />



```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```


<img width="239" height="35" alt="image" src="https://github.com/user-attachments/assets/ae41cf12-42e9-4790-98d3-9ac80de498da" />



```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```


<img width="325" height="38" alt="image" src="https://github.com/user-attachments/assets/ed2d6625-a4f5-4358-b5d3-5d3c64347318" />


```
data.shape
```


<img width="161" height="30" alt="image" src="https://github.com/user-attachments/assets/a08e654e-3928-4ca6-be03-38dc9773c9d4" />



```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data = {
    'Feature1': [1,2,3,4,5,],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}

df = pd.DataFrame(data)
x = df[['Feature1', 'Feature3']]
y = df[['Target']]

selector = SelectKBest(score_func=mutual_info_classif, k=1)
x_new = selector.fit_transform(x, y)

selected_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_feature_indices]

print("Selected Features:")
print(selected_features)
```


<img width="1781" height="92" alt="image" src="https://github.com/user-attachments/assets/dd8d9f3c-5861-4db0-8fe7-056bb82b6733" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="484" height="203" alt="image" src="https://github.com/user-attachments/assets/03939f97-69b3-4b3d-a3b6-ac4ee4808829" />


```
tips.time.unique()
```

<img width="439" height="58" alt="image" src="https://github.com/user-attachments/assets/802d35da-4c26-4158-981a-44f452c7fed4" />



```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```


<img width="290" height="98" alt="image" src="https://github.com/user-attachments/assets/601bab95-60b9-4a91-93e7-c3ae1781977f" />



```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```


<img width="408" height="47" alt="image" src="https://github.com/user-attachments/assets/2c6a654c-0f42-4792-a03f-b883dbeb92eb" />


       
# RESULT:
       # INCLUDE YOUR RESULT HERE
        Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.


