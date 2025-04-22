import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
df = pd.read_csv('features.csv')

# Drop filename
df = df.drop('filename', axis=1)

# Encode age labels
label_encoder = LabelEncoder()
df['age'] = label_encoder.fit_transform(df['age'])  # teen=2, adult=0, senior=1 (example)

# Separate features and target
X = df.drop('age', axis=1)
y = df['age']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model, scaler, and encoder                                                                  
joblib.dump(model, 'age_predictor_model.pkl')                                               
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

