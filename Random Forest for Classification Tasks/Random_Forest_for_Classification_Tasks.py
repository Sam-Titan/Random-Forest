import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

titanic_data = pd.read_csv("titanic.csv")
titanic_data = titanic_data.dropna(subset=['Survived'])

X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
X['Age'] = X['Age'].fillna(X['Age'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(n_estimators = 100, random_state = 21)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = model.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")

# 4. Improved Visualization
plt.figure(figsize=(10, 6))
# Plot correct vs incorrect to see model performance visually
correct = (y_test == y_pred)
plt.scatter(X_test.loc[correct, 'Age'], X_test.loc[correct, 'Fare'], 
            c='seagreen', label='Correct Prediction', alpha=0.6, edgecolors='w')
plt.scatter(X_test.loc[~correct, 'Age'], X_test.loc[~correct, 'Fare'], 
            c='crimson', label='Misclassified', marker='x', s=100)

plt.title('Random Forest Predictions: Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()