from flask import Flask, request, jsonify
from xgboost import Booster, DMatrix

app = Flask(__name__)
model = Booster()
model.load_model(fname=r"./xgb.model")


def volume_prediction(model, predictors):
    label = model.predict(DMatrix(predictors))
    return label[0]


@app.route('/')
def home():
    return "Hello there!"


@app.route('/predict', methods=['GET'])
def predict():

    vol_moving_avg = int(request.args.get("vol_moving_avg"))
    adj_close_rolling_med = int(request.args.get("adj_close_rolling_med"))

    predictors = [[vol_moving_avg, adj_close_rolling_med],
                  [vol_moving_avg, adj_close_rolling_med]]

    if not predictors:
        return jsonify({'error': 'You need to supply two values'}), 400

    return jsonify({
        "Volume Prediction": int(volume_prediction(model, predictors))
    })


if __name__ == "__main__":
    app.run()
