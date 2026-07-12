from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# تحميل ملفات الموديل
model = pickle.load(open("xgboost_final_model.pkl", "rb"))
feature_order = pickle.load(open("feature_order.pkl", "rb"))
street_map = pickle.load(open("street_encoding_map.pkl", "rb"))
global_mean = pickle.load(open("global_mean_price.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # ترتيب الميزات حسب feature_order
    features = [data[feat] for feat in feature_order]

    X = np.array([features])

    prediction = model.predict(X)[0]

    return jsonify({"prediction": float(prediction)})

@app.route('/', methods=['GET'])
def home():
    return "API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
