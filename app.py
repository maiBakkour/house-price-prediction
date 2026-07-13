import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'price_model.pkl'))
with open(os.path.join(BASE_DIR, 'model_maps.json'), encoding='utf-8') as f:
    maps = json.load(f)


def infer_apts_floor(area, is_front):
    """
    Rule-based fallback for apts_floor when the app doesn't send it.
    Raises ValueError if the area/facing combination is not a valid one.
    """
    if is_front:
        if 151 <= area <= 330:
            return 2
        elif 90 <= area <= 150:
            return 3
        elif 60 <= area < 90:
            raise ValueError(
                f"A front-facing unit of {area} sqm is not a valid combination "
                f"(front-facing units under 90 sqm are not supported)."
            )
        else:
            raise ValueError(
                f"No apts_floor rule defined for area={area} sqm, front-facing. "
                f"Please provide apts_floor explicitly."
            )
    else:
        if 90 <= area <= 150:
            return 3
        elif 60 <= area <= 89:
            return 4
        elif 151 <= area <= 330:
            raise ValueError(
                f"A back-facing unit of {area} sqm is not a valid combination "
                f"(back-facing units over 150 sqm are not supported)."
            )
        else:
            raise ValueError(
                f"No apts_floor rule defined for area={area} sqm, back-facing. "
                f"Please provide apts_floor explicitly."
            )


def build_feature_row(data):
    area = float(data['area'])
    rooms = int(data['rooms'])
    bathrooms = int(data['bathrooms'])
    floor = int(data['floor'])
    condition = data['condition']
    street = data['street']
    direction = str(data['direction']).strip().lower()
    is_front = bool(data['is_front'])

    if 'apts_floor' in data and data['apts_floor'] is not None:
        apts_floor = int(data['apts_floor'])
    else:
        apts_floor = infer_apts_floor(area, is_front)

    condition_num = maps['condition_map'].get(condition, 1)
    facing_num = 1 if is_front else 0
    street_enc = maps['street_encoded'].get(street, maps['global_mean_price'])

    direction_north = 1 if direction == 'north' else 0
    direction_west = 1 if direction == 'west' else 0

    row = {
        'الطابق': floor,
        'عدد الغرف': rooms,
        'عدد الحمامات': bathrooms,
        'المساحة (م2)': area,
        'الوجهة': facing_num,
        'الإكساء': condition_num,
        'عدد الشقق في كل طابق': apts_floor,
        'اتجاه_شمالي': direction_north,
        'اتجاه_غربي': direction_west,
        'الشارع_encoded': street_enc,
    }
    return pd.DataFrame([row])[maps['feature_order']], apts_floor


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df, apts_floor_used = build_feature_row(data)
        predicted = float(model.predict(input_df)[0])
        mae = maps['mae']

        return jsonify({
            'predicted_price': round(predicted),
            'min_price': round(predicted - mae),
            'max_price': round(predicted + mae),
            'apts_floor_used': apts_floor_used,
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'r2': maps['r2'],
        'mae': maps['mae'],
        'model_version': maps['model_version']
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
