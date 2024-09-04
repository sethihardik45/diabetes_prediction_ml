


# Diabetes Prediction Using Machine Learning

This project demonstrates how to use various machine learning algorithms to predict diabetes based on a dataset containing health-related features.

## Dataset

The dataset used in this project is the **diabetes.csv** file, which contains 768 data points, each with 9 features:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (target variable)

The `Outcome` feature indicates whether a patient has diabetes (`1`) or not (`0`).

## Exploratory Data Analysis

### 1. Checking the Dimensions
```python
print("Dimension of diabetes data: {}".format(diabetes.shape))
```
- Output: `dimension of diabetes data: (768, 9)`

### 2. Distribution of Outcome Variable
```python
import seaborn as sns
sns.countplot(diabetes['Outcome'], label="Count")
```

### 3. Dataset Information
```python
diabetes.info()
```

## K-Nearest Neighbors (KNN) to Predict Diabetes

The KNN algorithm is one of the simplest machine learning algorithms. It works by finding the closest data points in the training set to make predictions.

### Model Training and Evaluation
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.loc[:, diabetes.columns != 'Outcome'], 
    diabetes['Outcome'], 
    stratify=diabetes['Outcome'], 
    random_state=66
)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
```

- **Accuracy on Training Set:** 0.79
- **Accuracy on Test Set:** 0.78

## Decision Tree Classifier

### Initial Model
```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
```

- **Training Set Accuracy:** 1.000
- **Test Set Accuracy:** 0.714

### Applying Pre-Pruning (max_depth=3)
```python
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
```

- **Training Set Accuracy:** 0.773
- **Test Set Accuracy:** 0.740

### Feature Importance
```python
print("Feature importances:\n{}".format(tree.feature_importances_))

def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes.columns[:-1])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_diabetes(tree)
```

## Deep Learning to Predict Diabetes

### Training a Neural Network
```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
```

- **Training Set Accuracy:** 0.823
- **Test Set Accuracy:** 0.802

### Improving the Neural Network
```python
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
```

- **Training Set Accuracy:** 0.795
- **Test Set Accuracy:** 0.792

### Visualizing the First Layer Weights
```python
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(8), diabetes.columns[:-1])
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
```

## Conclusion

This project demonstrates how different machine learning algorithms, including K-Nearest Neighbors, Decision Trees, and Deep Learning, can be used to predict diabetes. Each model has its strengths and weaknesses, with neural networks showing potential for improvement with proper scaling and hyperparameter tuning.
