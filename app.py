import json
import pickle
import os
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import check_password_hash

app = Flask(__name__)

# Secret key untuk session/login.
# WAJIB di-set sebagai environment variable APP_SECRET di Railway.
APP_SECRET = os.getenv("APP_SECRET")
if not APP_SECRET:
    raise RuntimeError("APP_SECRET environment variable is not set")

app.config["SECRET_KEY"] = APP_SECRET
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True

MODEL_FILE = os.getenv("MODEL_FILE", "regmodel.pkl")
SCALER_FILE = os.getenv("SCALER_FILE", "scaling.pkl")

# Credentials login disimpan sebagai environment variables.
# ADMIN_PASSWORD_HASH harus berupa Werkzeug password hash.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

if not ADMIN_PASSWORD_HASH:
    raise RuntimeError("ADMIN_PASSWORD_HASH environment variable is not set")

scalar = pickle.load(open(SCALER_FILE, "rb"))
regmodel = pickle.load(open(MODEL_FILE, "rb"))


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if (
            username == ADMIN_USERNAME
            and check_password_hash(ADMIN_PASSWORD_HASH, password)
        ):
            session.clear()
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("home"))

        error = "Username atau password salah."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Endpoint root (home)
@app.route("/")
@login_required
def home():
    return render_template("home.html", username=session.get("username"))


# Endpoint prediksi harga rumah
@app.route("/predict_api", methods=["POST"])
@login_required
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))

    new_data = scalar.transform(
        np.array(list(data.values())).reshape(1, -1)
    )
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]

    return render_template(
        "home.html",
        username=session.get("username"),
        prediction_text=f"The House price prediction is {output}",
    )


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
