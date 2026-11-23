
import os
import io
import sqlite3
import secrets
import hashlib

from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path

import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    request,
    jsonify,
    session,
    redirect,
    send_from_directory,
    make_response
)

from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_fhm_model_finetuned.keras"
CLASS_NAMES_PATH = BASE_DIR / "fhm_class_names_large.txt"

IMG_SIZE = (128, 128)

CONFIDENCE_THRESHOLD = 0.55

DATABASE_PATH = BASE_DIR / "plant_purity.db"

FRONTEND_DIR = BASE_DIR / "frontend_v2"


app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "replace-this-key-before-deployment"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
)

CORS(app)


print("RUNNING BACKEND:", BASE_DIR)
print("DATABASE LOCATION:", DATABASE_PATH)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    with get_db() as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id)
            )
        """)

        db.commit()

        print("Database initialized successfully.")

        tables = db.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
        """).fetchall()

        print(
            "Available tables:",
            [row["name"] for row in tables]
        )


# IMPORTANT: Call initialization outside the function.
initialize_database()


# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(view):

    @wraps(view)
    def wrapped(*args, **kwargs):

        if "farmer_id" not in session:
            return jsonify({
                "error": "Please log in first."
            }), 401

        return view(*args, **kwargs)

    return wrapped


# ============================================================
# LOGIN PAGE
# ============================================================

@app.route("/login", methods=["GET"])
def login_page():

    return send_from_directory(BASE_DIR, "login.html")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home_page():

    if "farmer_id" not in session:
        return redirect("/login")

    html = (
        FRONTEND_DIR / "index.html"
    ).read_text(encoding="utf-8")

    html = html.replace(
        "</body>",
        '<script src="/auth-controls.js"></script>\n</body>'
    )

    return make_response(html)


@app.route("/index.html", methods=["GET"])
def index_page():

    if "farmer_id" not in session:
        return redirect("/login")

    html = (
        FRONTEND_DIR / "index.html"
    ).read_text(encoding="utf-8")

    html = html.replace(
        "</body>",
        '<script src="/auth-controls.js"></script>\n</body>'
    )

    response = make_response(html)

    response.headers["Cache-Control"] = "no-store"

    return response


# ============================================================
# FRONTEND FILES
# ============================================================

@app.route("/<path:filename>")
def frontend_files(filename):

    if filename in (
        "index.html",
        "model_performance.html"
    ):

        if "farmer_id" not in session:
            return redirect("/login")

        html = (
            FRONTEND_DIR / filename
        ).read_text(encoding="utf-8")

        html = html.replace(
            "</body>",
            '<script src="/auth-controls.js"></script>\n</body>'
        )

        response = make_response(html)

        response.headers["Cache-Control"] = "no-store"

        return response

    return send_from_directory(FRONTEND_DIR, filename)


# ============================================================
# JAVASCRIPT FILES
# ============================================================

@app.route("/auth.js", methods=["GET"])
def auth_javascript():

    return send_from_directory(BASE_DIR, "auth.js")


@app.route("/auth-controls.js", methods=["GET"])
def auth_controls_javascript():

    return send_from_directory(BASE_DIR, "auth-controls.js")


# ============================================================
# FARMER REGISTRATION
# ============================================================

@app.route("/auth/register", methods=["POST"])
def register_farmer():

    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or not password:

        return jsonify({
            "error": "Please fill in all fields."
        }), 400

    if len(password) < 8:

        return jsonify({
            "error": "Password must be at least 8 characters."
        }), 400

    password_hash = generate_password_hash(password)

    try:

        with get_db() as db:

            cursor = db.execute(
                """
                INSERT INTO farmers
                (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (name, email, password_hash)
            )

            farmer_id = cursor.lastrowid

    except sqlite3.IntegrityError:

        return jsonify({
            "error": "This email is already registered. Please log in."
        }), 409

    session.clear()

    session.permanent = True

    session["farmer_id"] = farmer_id
    session["farmer_name"] = name

    return jsonify({
        "message": "Account created.",
        "redirect": "/"
    }), 201


# ============================================================
# FARMER LOGIN
# ============================================================

@app.route("/auth/login", methods=["POST"])
def login_farmer():

    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:

        return jsonify({
            "error": "Enter your email and password."
        }), 400

    with get_db() as db:

        farmer = db.execute(
            """
            SELECT id, name, password_hash
            FROM farmers
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if (
        farmer is None
        or not check_password_hash(
            farmer["password_hash"],
            password
        )
    ):

        return jsonify({
            "error": "Incorrect email or password."
        }), 401

    session.clear()

    session.permanent = True

    session["farmer_id"] = farmer["id"]
    session["farmer_name"] = farmer["name"]

    return jsonify({
        "message": "Login successful.",
        "redirect": "/"
    }), 200


# ============================================================
# FARMER LOGOUT
# ============================================================

@app.route("/auth/logout", methods=["POST"])
def logout_farmer():

    session.clear()

    return jsonify({
        "message": "Logged out.",
        "redirect": "/login"
    }), 200


# ============================================================
# FORGOT PASSWORD
# LOCAL DEMONSTRATION VERSION
# ============================================================

@app.route("/auth/forgot-password", methods=["POST"])
def forgot_password():

    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()

    if not email:

        return jsonify({
            "error": "Please enter your registered email."
        }), 400

    with get_db() as db:

        print("DATABASE USED:", DATABASE_PATH)

        tables = db.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
        """).fetchall()

        print(
            "AVAILABLE TABLES:",
            [row["name"] for row in tables]
        )

        farmer = db.execute(
            """
            SELECT id
            FROM farmers
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        if farmer is None:

            return jsonify({
                "message": "If the account exists, a reset link will be generated."
            }), 200

        # Generate secure random token
        token = secrets.token_urlsafe(32)

        # Hash the token before storing it
        token_hash = hashlib.sha256(
            token.encode()
        ).hexdigest()

        # Token expires after 30 minutes
        expires_at = (
            datetime.now(timezone.utc)
            + timedelta(minutes=30)
        ).isoformat()

        db.execute("""
            INSERT INTO password_reset_tokens
            (
                farmer_id,
                token_hash,
                expires_at,
                used
            )
            VALUES (?, ?, ?, 0)
        """, (
            farmer["id"],
            token_hash,
            expires_at
        ))

    # Generate local reset URL
    reset_url = (
        request.host_url.rstrip("/")
        + "/reset-password?token="
        + token
    )

    # Display link in Flask terminal
    print("\n" + "=" * 60)
    print("LOCAL PASSWORD RESET LINK")
    print(reset_url)
    print("Link expires in 30 minutes.")
    print("=" * 60 + "\n")

    return jsonify({
        "message": "Reset link generated successfully. Open the link below.",
        "reset_url": reset_url
    }), 200


# ============================================================
# RESET PASSWORD PAGE
# ============================================================


@app.route("/reset-password", methods=["GET"])
def reset_password_page():
    return send_from_directory(
        FRONTEND_DIR,
        "reset_password.html"
    )



# ============================================================
# RESET PASSWORD API
# ============================================================

@app.route("/auth/reset-password", methods=["POST"])
def reset_password():

    data = request.get_json(silent=True) or {}

    token = str(data.get("token", "")).strip()

    new_password = str(data.get("password", ""))

    if not token or not new_password:

        return jsonify({
            "error": "Reset token and new password are required."
        }), 400

    if len(new_password) < 8:

        return jsonify({
            "error": "Password must be at least 8 characters."
        }), 400

    token_hash = hashlib.sha256(
        token.encode()
    ).hexdigest()

    now = datetime.now(timezone.utc)

    with get_db() as db:

        reset_record = db.execute("""
            SELECT
                id,
                farmer_id,
                expires_at,
                used
            FROM password_reset_tokens
            WHERE token_hash = ?
        """, (token_hash,)).fetchone()

        if reset_record is None:

            return jsonify({
                "error": "Invalid password reset link."
            }), 400

        if reset_record["used"]:

            return jsonify({
                "error": "This reset link has already been used."
            }), 400

        expiry = datetime.fromisoformat(
            reset_record["expires_at"]
        )

        if now >= expiry:

            return jsonify({
                "error": "This reset link has expired. Please request a new one."
            }), 400

        new_password_hash = generate_password_hash(
            new_password
        )

        db.execute("""
            UPDATE farmers
            SET password_hash = ?
            WHERE id = ?
        """, (
            new_password_hash,
            reset_record["farmer_id"]
        ))

        db.execute("""
            UPDATE password_reset_tokens
            SET used = 1
            WHERE id = ?
        """, (
            reset_record["id"],
        ))

    return jsonify({
        "message": "Password changed successfully. Please log in.",
        "redirect": "/login"
    }), 200


# ============================================================
# LOAD MODEL AND CLASS NAMES
# ============================================================

print("Loading model, please wait...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    CLASS_NAMES = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("Model loaded.")
print("Classes:", CLASS_NAMES)


preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


# ============================================================
# VEGETATION PRESENCE CHECK
# ============================================================

def check_vegetation_presence(
    pil_image,
    min_green_ratio=0.005
):

    img = pil_image.convert("RGB").resize((150, 150))

    arr = np.array(
        img,
        dtype=np.float32
    )

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    max_val = np.maximum(
        np.maximum(r, g),
        b
    )

    min_val = np.minimum(
        np.minimum(r, g),
        b
    )

    delta = max_val - min_val + 1e-6

    hue = np.zeros_like(max_val)

    mask_r = max_val == r
    mask_g = max_val == g
    mask_b = max_val == b

    hue[mask_r] = (
        (g[mask_r] - b[mask_r])
        / delta[mask_r]
    ) % 6

    hue[mask_g] = (
        (b[mask_g] - r[mask_g])
        / delta[mask_g]
    ) + 2

    hue[mask_b] = (
        (r[mask_b] - g[mask_b])
        / delta[mask_b]
    ) + 4

    hue = (hue * 60.0) % 360.0

    sat = np.where(
        max_val > 0,
        delta / max_val * 100.0,
        0
    )

    val = (max_val / 255.0) * 100.0

    foliage_mask = (
        (hue >= 50)
        & (hue <= 170)
        & (sat >= 8)
        & (val >= 8)
    )

    green_ratio = float(
        np.mean(foliage_mask)
    )

    return (
        green_ratio >= min_green_ratio,
        green_ratio
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_image):

    img = pil_image.convert("RGB").resize(
        IMG_SIZE
    )

    arr = np.array(
        img,
        dtype=np.float32
    )

    arr = preprocess_input(arr)

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


# ============================================================
# PLANT PURITY PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No image file provided. Send it as form-data with key 'file'."
            }), 400

        file = request.files["file"]

        image_bytes = file.read()

        try:

            pil_image = Image.open(
                io.BytesIO(image_bytes)
            )

        except Exception as e:

            return jsonify({
                "error": f"Could not read image file: {e}"
            }), 400

        # Check vegetation presence
        has_vegetation, green_ratio = (
            check_vegetation_presence(pil_image)
        )

        if not has_vegetation:

            return jsonify({

                "is_seedling_image": False,

                "prediction": None,

                "confidence": None,

                "green_pixel_ratio": (
                    f"{green_ratio * 100:.1f}%"
                ),

                "note": (
                    "This does not appear to be a maize seedling image. "
                    "The photo does not contain enough plant-like green colouring. "
                    "This is a basic colour heuristic, not a trained model. "
                    "Please upload a clear photo of a germinated seedling."
                )

            })

        # Preprocess image
        processed = preprocess_image(
            pil_image
        )

        # Model prediction
        predictions = model.predict(
            processed,
            verbose=0
        )[0]

        predicted_idx = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[predicted_idx]
        )

        predicted_class = CLASS_NAMES[
            predicted_idx
        ]

        is_confident = (
            confidence >= CONFIDENCE_THRESHOLD
        )

        response = {

            "is_seedling_image": True,

            "prediction": (
                predicted_class
                if is_confident
                else "Uncertain"
            ),

            "confidence": (
                f"{confidence * 100:.1f}%"
            ),

            "confident": is_confident,

            "all_class_probabilities": {

                CLASS_NAMES[i]: (
                    f"{float(predictions[i]) * 100:.1f}%"
                )

                for i in range(len(CLASS_NAMES))

            }

        }

        if not is_confident:

            response["note"] = (
                f"Model confidence ({confidence * 100:.1f}%) "
                f"is below the {CONFIDENCE_THRESHOLD * 100:.0f}% threshold. "
                "Consider uploading a clearer image."
            )

        return jsonify(response)

    except Exception as e:

        app.logger.exception(
            "Prediction failed"
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({

        "status": "active",

        "model_loaded": True,

        "classes": CLASS_NAMES,

        "confidence_threshold": CONFIDENCE_THRESHOLD

    })


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("PLANT PURITY PREDICTION API")

    print("=" * 60)

    print("Model:", MODEL_PATH)

    print("Classes:", CLASS_NAMES)

    print(
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD
    )

    print(
        "Starting server at http://127.0.0.1:5000"
    )

    print("=" * 60 + "\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
