import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target

X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

single_data = X_test.iloc[[0]]
predicted_value = model.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")

print(f"Mean Squared Error: {MSE:.2f}")
print(f"Mean Absolute Error: {MAE:.2f}")
print(f"R-squared Score: {R2:.2f}")

plt.figure(figsize=(10, 6))

# 1. Scatter plot of Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.3, color='blue', label='Predictions')

# 2. The "Perfect Prediction" reference line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')

# 3. Adding the Metrics Text Box
# We use 'axes fraction' so the text stays in the same spot regardless of data scale
stats_text = (f'MSE: {MSE:.2f}\n'f'MAE: {MAE:.2f}\n'f'R² Score: {R2:.2f}')
# This puts the box in the upper left (0.05, 0.95)
plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
               fontsize=12, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
plt.xlabel('Actual Values ($100k)')
plt.ylabel('Predicted Values ($100k)')
plt.title('Random Forest Regression: California Housing')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 2. Feature Importance Plot
importances = model.feature_importances_
features = X.columns
# Sort indices to show most important features at the top
import_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(import_df['Feature'], import_df['Importance'], color='skyblue')
plt.xlabel('Relative Importance')
plt.title('Feature Importances in Random Forest')
plt.tight_layout()
plt.show()