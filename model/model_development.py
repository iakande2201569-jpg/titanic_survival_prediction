import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Load the Titanic dataset
df = pd.read_csv('train.csv')

# 2. Feature Selection (Selecting 5 features)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
target = 'Survived'

X = df[features]
y = df[target]

# 3. Preprocessing
numeric_features = ['Age', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex'] # Pclass is numeric but acts categorical

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Implement Random Forest Classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Model
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model, 'model/titanic_survival_model.pkl')
print("Model saved to model/titanic_survival_model.pkl")