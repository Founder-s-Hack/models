import firebase_admin
from firebase_admin import credentials, storage
import joblib
import os
import tempfile

def initialize_firebase():
    # Use your Firebase project's credentials JSON file
    cred = credentials.Certificate("path/to/your/firebase/credentials.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'your-bucket-name.appspot.com'
    })

def load_model():
    initialize_firebase()
    bucket = storage.bucket()
    blob = bucket.blob('path/to/your/model.pkl')
    _, temp_local_filename = tempfile.mkstemp()
    blob.download_to_filename(temp_local_filename)
    model = joblib.load(temp_local_filename)
    os.remove(temp_local_filename)
    return model

def run_model(model, data):
    # Assuming the model is a scikit-learn model
    # Extract input features from data
    loan_amount = data['loanAmount']
    org_employee_count = data['orgEmployeeCount']
    org_is_new = data['orgIsNew']
    org_anzsic = data['orgANZSIC']
    
    # Create an input array for the model
    input_data = [[loan_amount, org_employee_count, org_is_new, org_anzsic]]
    
    # Run the model prediction
    prediction = model.predict(input_data)
    
    # Example response structure
    response = {
        'approved': bool(prediction[0]),
        'interestRate': 5.5 if prediction[0] else None
    }
    return response