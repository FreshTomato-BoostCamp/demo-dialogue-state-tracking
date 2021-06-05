import os
import sys
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
from config import CFG
from model import TRADE

app = Flask(__name__)
app.config.from_object(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["POST", "GET"])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

@app.route("/success/<name>", methods=["GET"])
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "


def initialize_model():
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
