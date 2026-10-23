# All Codes and Outputs

## 1. Data Preprocessing

### Code

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data (1).csv")

# Display first 5 rows
print("Original Dataset:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values (if any)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select numerical columns
X = data[['Hours_Studied', 'Attendance', 'Assignments', 'Marks']]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_Scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Display normalized dataset
print("\nNormalized Dataset:")
print(X_Scaled.head())










import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
data = pd.read_csv("data.csv")
# Features (Input)
X = data[['Hours_Studied','Attendance','Assignments']]
# Target (Output)
y = data['Result']
# Split data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Display sizes
print("Training Data:")
print(X_train)
print("\nTesting Data:")
print(X_test)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
```

### Output

```text
Original Dataset:
   StudentID  Hours_Studied  Attendance  Assignments  Marks Result
0          1              2          75            3     45   Fail
1          2              3          80            4     52   Pass
2          3              4          85            5     60   Pass
3          4              1          65            2     35   Fail
4          5              5          90            5     75   Pass

Missing Values:
StudentID        0
Hours_Studied    0
Attendance       0
Assignments      0
Marks            0
Result           0

Normalized Dataset:
   Hours_Studied  Attendance  Assignments     Marks
0      -0.958215   -0.697680    -0.912871 -0.965884
1      -0.440261   -0.267013     0.000000 -0.603354
2       0.077693    0.163653     0.912871 -0.189034
3      -1.476169   -1.559013    -1.825742 -1.483784
4       0.595647    0.594320     0.912871  0.587817
```

## 2. Data Splitting

### Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
data = pd.read_csv("data.csv")
# Features (Input)
X = data[['Hours_Studied', 'Attendance', 'Assignments']]
# Target (Output)
y = data['Result']
# Split data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Display sizes
print("Training Data:")
print(X_train)
print("\nTesting Data:")
print(X_test)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
```

### Output

```text
Training Data:
    Hours_Studied  Attendance  Assignments
8               3          78            4
5               6          95            5
11              5          92            5
3               1          65            2
18              5          91            5
16              7          99            5
13              2          72            3
2               4          85            5
9               4          88            4
19              6          97            5
4               5          90            5
12              6          96            5
7               7          98            5
10              1          60            2
14              4          84            4
6               2          70            3

Testing Data:
    Hours_Studied  Attendance  Assignments
0               2          75            3
17              1          68            2
15              3          79            4
1               3          80            4

Training Samples: 16
Testing Samples: 4
```

## 3. Cross Validation

### Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

# Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Perform 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean())
```

### Output

```text
Training Data Size: 16
Testing Data Size: 4

Cross Validation Scores:
[0.75 1.   1.   1.   1.  ]

Average Accuracy:
0.95
```

## 4. Linear Regression

### Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")

# Input feature
X = data[['Hours_Studied']]

# Target variable
y = data['Marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Evaluate performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Display predictions
print("\nPredicted Marks:")
print(y_pred)

# Graph
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.legend()
plt.show()
```

### Output

```text
Mean Squared Error: 6.750000000000021
R2 Score: 0.879866518353726

Predicted Marks:
[45. 35. 55. 55.]
```

### Graphs

- [linear_regression.png](linear_regression.png)

## 5. KNN Classification

### Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load dataset
data = pd.read_csv("data.csv")
# Input features
X = data[['Hours_Studied', 'Attendance','Assignments']]
# Target variable
y = data['Result']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)
# Train model
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Output

```text
Accuracy: 1.0

Confusion Matrix:
[[2 0]
 [0 2]]

Classification Report:
              precision    recall  f1-score   support

        Fail       1.00      1.00      1.00         2
        Pass       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

## 6. SVM Classification

### Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv("data.csv")

# Input features
X = data[['Hours_Studied', 'Attendance', 'Assignments']]

# Target variable
y = data['Result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv("data.csv")

# Input features
X = data[['Hours_Studied', 'Attendance', 'Assignments']]

# Target variable
y = data['Result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 1. Confusion Matrix Graph
# -----------------------------
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")
plt.title("SVM Confusion Matrix")
plt.show()


# -----------------------------
# 2. Evaluation Metrics Graph
# -----------------------------
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

metrics = {
    "Accuracy": accuracy,
    "Precision": report["weighted avg"]["precision"],
    "Recall": report["weighted avg"]["recall"],
    "F1-Score": report["weighted avg"]["f1-score"]
}

plt.figure(figsize=(7, 5))

plt.bar(
    metrics.keys(),
    metrics.values(),
    color=['blue', 'green', 'orange', 'red']
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("SVM Classification Evaluation Metrics")

for i, value in enumerate(metrics.values()):
    plt.text(i, value + 0.02, f"{value:.2f}", ha='center')

plt.show()
```

### Output

```text
Accuracy: 1.0

Confusion Matrix:
[[2 0]
 [0 2]]

Classification Report:
              precision    recall  f1-score   support

        Fail       1.00      1.00      1.00         2
        Pass       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

## 7. K-Means Clustering

### Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load dataset
data = pd.read_csv("data.csv")
# Select features
X = data[['Hours_Studied','Marks']]
# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit the model
kmeans.fit(X)
# Predict cluster labels
clusters = kmeans.labels_
# Add cluster column to dataset
data['Cluster'] = clusters
# Display clustered data
print(data)
# Plot clusters
plt.scatter(X['Hours_Studied'], X['Marks'], c=clusters)
# Plot centroids
plt.scatter(
kmeans.cluster_centers_[:, 0],
kmeans.cluster_centers_[:, 1],
marker='X',
s=200,
label='Centroids'
)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
```

### Output

```text
    StudentID  Hours_Studied  Attendance  Assignments  Marks Result  Cluster
0           1              2          75            3     45   Fail        0
1           2              3          80            4     52   Pass        1
2           3              4          85            5     60   Pass        1
3           4              1          65            2     35   Fail        0
4           5              5          90            5     75   Pass        2
5           6              6          95            5     82   Pass        2
6           7              2          70            3     48   Fail        0
7           8              7          98            5     90   Pass        2
8           9              3          78            4     55   Pass        1
9          10              4          88            4     68   Pass        1
10         11              1          60            2     30   Fail        0
11         12              5          92            5     80   Pass        2
12         13              6          96            5     85   Pass        2
13         14              2          72            3     46   Fail        0
14         15              4          84            4     65   Pass        1
15         16              3          79            4     58   Pass        1
16         17              7          99            5     95   Pass        2
17         18              1          68            2     38   Fail        0
18         19              5          91            5     78   Pass        2
19         20              6          97            5     88   Pass        2
```

### Graphs

- [kmeans.png](kmeans.png)

## 8. Hierarchical Clustering

### Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
# Load dataset
data = pd.read_csv("data.csv")
# Select features
X = data[['Hours_Studied','Marks']]
# Create linkage matrix
linked = linkage(X,method='ward')
# Plot Dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()
# Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)
clusters = hc.fit_predict(X)
# Add cluster labels to dataset
data['Cluster'] = clusters
print(data)
# Scatter Plot
plt.scatter(X['Hours_Studied'], X['Marks'], c=clusters)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hierarchical Clustering")
plt.show()
```

### Output

```text
    StudentID  Hours_Studied  Attendance  Assignments  Marks Result  Cluster
0           1              2          75            3     45   Fail        1
1           2              3          80            4     52   Pass        2
2           3              4          85            5     60   Pass        2
3           4              1          65            2     35   Fail        1
4           5              5          90            5     75   Pass        0
5           6              6          95            5     82   Pass        0
6           7              2          70            3     48   Fail        1
7           8              7          98            5     90   Pass        0
8           9              3          78            4     55   Pass        2
9          10              4          88            4     68   Pass        2
10         11              1          60            2     30   Fail        1
11         12              5          92            5     80   Pass        0
12         13              6          96            5     85   Pass        0
13         14              2          72            3     46   Fail        1
14         15              4          84            4     65   Pass        2
15         16              3          79            4     58   Pass        2
16         17              7          99            5     95   Pass        0
17         18              1          68            2     38   Fail        1
18         19              5          91            5     78   Pass        0
19         20              6          97            5     88   Pass        0
```

### Graphs

- [dendrogram.png](dendrogram.png)
- [hierarchical_clustering.png](hierarchical_clustering.png)

## 9. Random Forest

### Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv("data.csv")
# Input features
X = data[['Hours_Studied','Attendance','Assignments']]
# Target variable
y = data['Result']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create Random Forest model
model = RandomForestClassifier(n_estimators=100,random_state=42)
# Train the model
model.fit(X_train, y_train)
# Predict test data
y_pred = model.predict(X_test)
# Evaluate performance
print("Accuracy", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Feature Importance Graph
importance = model.feature_importances_
plt.bar(X.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()
```

### Output

```text
Accuracy: 1.0

Confusion Matrix:
[[2 0]
 [0 2]]

Classification Report:
              precision    recall  f1-score   support

        Fail       1.00      1.00      1.00         2
        Pass       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

### Graphs

- [random_forest_feature_importance.png](random_forest_feature_importance.png)

## 10. AdaBoost

### Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

importance = model.feature_importances_
plt.bar(X.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("AdaBoost Feature Importance")
plt.show()
```

### Output

```text
Accuracy: 1.0

Confusion Matrix:
[[2 0]
 [0 2]]

Classification Report:
              precision    recall  f1-score   support

        Fail       1.00      1.00      1.00         2
        Pass       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

### Graphs

- [adaboost_feature_importance.png](adaboost_feature_importance.png)

## 11. Q-Learning

### Code

```python
import numpy as np

R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

Q = np.zeros_like(R, dtype=float)

gamma = 0.8
episodes = 1000

for _ in range(episodes):
    state = np.random.randint(0, 6)

    while state != 5:
        actions = np.where(R[state] >= 0)[0]
        action = np.random.choice(actions)
        next_state = action

        Q[state, action] = (
            R[state, action] + gamma * np.max(Q[next_state])
        )

        state = next_state

print("Q-Table:\n")
print(Q)
```

### Output

```text
Q-Table:

[[  0.    0.    0.    0.   80.    0. ]
 [  0.    0.    0.   64.    0.  100. ]
 [  0.    0.    0.   64.    0.    0. ]
 [  0.   80.   51.2   0.   80.    0. ]
 [ 64.    0.    0.   64.    0.  100. ]
 [  0.    0.    0.    0.    0.    0. ]]
```

## 12. Neural Network

### Code

```python
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")

X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

y = y.map({'Fail': 0, 'Pass': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).ravel()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Output

```text
TensorFlow output was not generated in this environment. Run neural_network.py locally to produce the final output.
```


