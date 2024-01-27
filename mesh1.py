
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
df = pd.read_csv('output_csv_file.csv')

# Split the dataset into features (X) and target (y)
X = df[['Force (N)']]
y = df['Materials']

# Encode the categorical labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Create and train the model (Decision Tree in this example)
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Save the trained model to a file
joblib.dump(model, 'material_prediction_model.joblib')

# Now, you can use the saved model to predict materials for new force values
# Load the saved model
loaded_model = joblib.load('material_prediction_model.joblib')

# Get force value from the user
user_input_force = float(input('Enter Force (N): '))

# Check if the force is exactly 0 before making a prediction
if user_input_force == 0:
    print('No material prediction for zero force.')
else:
    # Make a prediction using the loaded model
    user_input_data = pd.DataFrame({'Force (N)': [user_input_force]})
    predicted_material_encoded = loaded_model.predict(user_input_data)

    # Decode the numerical prediction back to the original label
    predicted_material_label = le.inverse_transform(predicted_material_encoded)

    print(f'Predicted Material: {predicted_material_label[0]}')
