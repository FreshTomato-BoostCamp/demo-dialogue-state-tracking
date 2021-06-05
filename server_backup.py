import os
import sys
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
from config import CFG
from model import TRADE

app = Flask(__name__)
app.config.from_object(__name__)

@app.route("/")
def index():
    initialize_model()
    return redirect(url_for("home"))

# /html/cvr.html
@app.route("/html/cvr.html", methods=["GET", "POST", "PUT"])
def home():
    return render_template("home.html")

# /get/cvr
@app.route("/state", methods=["GET"])
def inference():
    line = int(request.args['inputLine'])
    X = dataset[line_iloc]["data"]
    X = apply_preprocessing(X.reshape(1, -1), stats, encoder, verbose=False)
    prob = model.predict_proba(X).item()
    result = f"{prob*100:.4f}%"
    return render_template("result.html", value=line, result=result)


@app.route("/update/model", methods=["PUT"])
def model():
    global model_version
    compare_version = get_model_version(Config.BestModelSavePath)
    if model_version == compare_version:
        result = "이미 최신 버전의 모델입니다😊"
    else:
        initialize_model()
        result = "모델이 업데이트 되었습니다!👍"
    return render_template("update.html", result=result)


def initialize_model():
    model = 
    
    global stats, model_loaded, model, encoder, model_version
    model_dat = load_model_dat(Config.BestModelSavePath)
    model_version = save_model_version(Config.BestModelSavePath, save_path="./server/")

    stats = model_dat["stats"]
    model_type = model_dat["model_type"]
    weights = model_dat["weights"]
    model = get_model(model_type)
    model.load_from_weights(weights)
    encoder = IntegratedEncoder(Config.Encoders)
    model_loaded = True


if __name__ == "__main__":
    app.run(debug=True)
