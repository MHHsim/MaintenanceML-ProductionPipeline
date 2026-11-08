# -*- coding: utf-8 -*-
"""
API de mantenimiento predictivo.

Carga el pipeline entrenado en la Fase 7 (preprocesado + modelo LightGBM) y
expone dos formas de uso:

- Un formulario web (GET/POST "/") para introducir los datos de una máquina y
  ver si se predice fallo.
- Una API JSON (POST "/api/predict") para integrarlo con otros sistemas.

La predicción aplica el umbral de decisión guardado en los metadatos (0.1547),
elegido en la Fase 7 priorizando el recall (detectar los fallos reales).
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# El pipeline avisa de nombres de features al pasar de ColumnTransformer; es inocuo
warnings.filterwarnings("ignore")

# CORS es opcional: si no está instalado, la app funciona igual en local
try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

# ---------------------------------------------------------------------------
# Carga del modelo y los metadatos (una sola vez, al arrancar)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_mantenimiento.joblib"
META_PATH = BASE_DIR / "models" / "modelo_mantenimiento_metadata.json"

modelo = joblib.load(MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    META = json.load(f)

FEATURES = META["features_entrada"]        # orden de columnas que espera el modelo
UMBRAL = float(META["umbral_decision"])    # umbral de decisión (0.1547)
TIPOS_VALIDOS = {"L", "M", "H"}            # categorías válidas de Type

# Variables numéricas = todas las features menos "Type"
FEATURES_NUM = [f for f in FEATURES if f != "Type"]

app = Flask(__name__)
if _HAS_CORS:
    CORS(app)


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------
def validar_y_preparar(datos: dict) -> pd.DataFrame:
    """Valida el diccionario de entrada y lo convierte en un DataFrame de 1 fila.

    Lanza ValueError con un mensaje claro si algo no cuadra.
    """
    fila = {}

    # Type: categórica, debe ser L, M o H
    tipo = str(datos.get("Type", "")).strip().upper()
    if tipo not in TIPOS_VALIDOS:
        raise ValueError("'Type' debe ser 'L', 'M' o 'H'.")
    fila["Type"] = tipo

    # Variables numéricas: deben existir y ser convertibles a número
    for col in FEATURES_NUM:
        valor = datos.get(col, None)
        if valor is None or str(valor).strip() == "":
            raise ValueError(f"Falta el valor de '{col}'.")
        try:
            fila[col] = float(valor)
        except (TypeError, ValueError):
            raise ValueError(f"'{col}' debe ser un número.")

    # DataFrame con las columnas en el orden exacto que espera el pipeline
    return pd.DataFrame([fila])[FEATURES]


def predecir(datos: dict) -> dict:
    """Devuelve la predicción para un diccionario de datos de una máquina."""
    X = validar_y_preparar(datos)
    prob = float(modelo.predict_proba(X)[0, 1])
    falla = prob >= UMBRAL
    return {
        "prediccion": int(falla),
        "etiqueta": "Fallo probable" if falla else "Funcionamiento normal",
        "probabilidad_fallo": round(prob, 4),
        "umbral": UMBRAL,
    }


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Formulario web: muestra el formulario y, al enviarlo, el resultado."""
    resultado = None
    error = None
    valores = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            resultado = predecir(valores)
        except ValueError as e:
            error = str(e)

    return render_template(
        "index.html",
        features_num=FEATURES_NUM,
        tipos=sorted(TIPOS_VALIDOS),
        resultado=resultado,
        error=error,
        valores=valores,
        umbral=UMBRAL,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API JSON: recibe un JSON con las features y devuelve la predicción."""
    datos = request.get_json(silent=True)
    if datos is None:
        return jsonify({"error": "Envía un cuerpo JSON con las variables de la máquina."}), 400
    try:
        return jsonify(predecir(datos))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Comprobación de estado para el despliegue."""
    return jsonify({
        "status": "ok",
        "modelo": META.get("modelo"),
        "umbral": UMBRAL,
        "features": FEATURES,
    })


if __name__ == "__main__":
    # Servidor de desarrollo. En producción se sirve con gunicorn (ver Procfile).
    app.run(host="0.0.0.0", port=5000, debug=True)
