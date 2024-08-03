from flask import Flask, request, jsonify
from model_utils import load_model, run_model

app = Flask(__name__)

# Load the model from Firebase
model = load_model()

@app.route('/lending/request-loan', methods=['POST'])
def request_loan():
    data = request.get_json()
    # Run the model with the input data
    result = run_model(model, data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969)
