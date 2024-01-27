

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

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

# Streamlit UI
st.title('Material Prediction App')

# Get force value from the user
user_input_force = st.number_input('Enter Force (N):')

if user_input_force == 0:
    st.warning('Please enter a force value greater than zero to make a prediction.')
else:
    # Make a prediction using the loaded model
    user_input_data = pd.DataFrame({'Force (N)': [user_input_force]})
    predicted_material_encoded = loaded_model.predict(user_input_data)

    # Decode the numerical prediction back to the original label
    predicted_material_label = le.inverse_transform(predicted_material_encoded)

    # Get additional information based on the predicted material
    predicted_material_info = df[df['Materials'] == predicted_material_label[0]].iloc[0]

    # Display the prediction result along with additional information
    st.write(f'Predicted Material: {predicted_material_label[0]}')
    st.write(f'Infill%: {predicted_material_info["Infill%"]}')
    st.write(f'Infill Type: {predicted_material_info["Infill Type"]}')
    st.write(f'Mesh Type: {predicted_material_info["Mesh Type"]}')
    st.write(f'Number of Mesh Layers: {predicted_material_info["Number of mesh layer"]}')
    st.write(f'Base Material: {predicted_material_info["Base Material"]}')




