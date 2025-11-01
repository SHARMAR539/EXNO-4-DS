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
<img width="1357" height="425" alt="image" src="https://github.com/user-attachments/assets/981269b2-94a6-44e6-a012-2cd0af5750fd" />


```
data.isnull().sum()
```

<img width="159" height="493" alt="image" src="https://github.com/user-attachments/assets/c44b43a8-be95-4a2a-bcd6-913649dad540" />

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1372" height="403" alt="image" src="https://github.com/user-attachments/assets/3e0fb829-958d-43ea-a0ad-3c02c04d635a" />


```
data2=data.dropna(axis=0)
data2
```

<img width="1403" height="424" alt="image" src="https://github.com/user-attachments/assets/59d69a14-9d0a-477c-9353-459b139c5acf" />


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="1135" height="330" alt="image" src="https://github.com/user-attachments/assets/7ea5ef18-4c67-4142-a7bc-f9e96bdc6eae" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="440" height="429" alt="image" src="https://github.com/user-attachments/assets/3b5126ed-a135-421f-aea2-ba18bbcc12f8" />


```
data2
```


<img width="1324" height="419" alt="image" src="https://github.com/user-attachments/assets/4848ba93-4b11-472e-9d4b-a397a7d8f14d" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```


<img width="1492" height="507" alt="image" src="https://github.com/user-attachments/assets/1b249206-e243-4c91-ac7e-3d09ac2a639b" />


```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1452" height="48" alt="image" src="https://github.com/user-attachments/assets/8fd77d88-e5fc-4ff6-b9a4-1a1c9947580f" />


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1448" height="39" alt="image" src="https://github.com/user-attachments/assets/e626228f-b4d9-4f1f-90c9-1bd13b95a1c6" />


```
y=new_data['SalStat'].values
print(y)
```

<img width="239" height="36" alt="image" src="https://github.com/user-attachments/assets/112feffb-6815-4c43-9335-115d0d782bff" />


```
x=new_data[features].values
print(x)
```

<img width="414" height="135" alt="image" src="https://github.com/user-attachments/assets/1a5a341c-53c2-41ff-8dc5-f718d1ffc2e8" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size = 0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```

<img width="325" height="79" alt="image" src="https://github.com/user-attachments/assets/9387b711-c737-4a2f-89bc-963ecd5ee30f" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="212" height="61" alt="image" src="https://github.com/user-attachments/assets/2abe3f21-7d77-4e1c-9b00-1acb6cfeb89d" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```


<img width="219" height="43" alt="image" src="https://github.com/user-attachments/assets/0fd3a493-198e-437b-a271-2997fd56e10b" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="297" height="47" alt="image" src="https://github.com/user-attachments/assets/357fd6d3-4d42-42fa-bbd3-3ba085a59f69" />


```
data.shape
```


<img width="176" height="34" alt="image" src="https://github.com/user-attachments/assets/52f6fde0-f190-458d-a286-0e26448ed5ed" />


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


<img width="1473" height="101" alt="image" src="https://github.com/user-attachments/assets/f78a13c9-276d-4c69-92e8-da2f7997c4b3" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="537" height="260" alt="image" src="https://github.com/user-attachments/assets/18bb18f1-d50d-40a6-afcb-7866ac959491" />


```
tips.time.unique()
```

<img width="404" height="75" alt="image" src="https://github.com/user-attachments/assets/a864569a-7382-4ddb-b8c4-d69c3482f881" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="260" height="99" alt="image" src="https://github.com/user-attachments/assets/b0dd6384-920e-4f11-b76a-c59ab6567240" />




```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="435" height="59" alt="image" src="https://github.com/user-attachments/assets/115deff1-07d4-4b2d-abfd-4162e20e7756" />















# RESULT:
       # INCLUDE YOUR RESULT HERE
