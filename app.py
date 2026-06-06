from flask import Flask, render_template, request, jsonify
from analyzer import analyze_face_set, REQUIRED_ANGLES

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html", required_angles=REQUIRED_ANGLES)

@app.post("/api/analyze")
def analyze():
    """
    Expects multipart form-data fields:
      - front
      - up
      - down
      - left
      - right
    """
    payload = {}
    for angle in REQUIRED_ANGLES:
        f = request.files.get(angle)
        if f and f.filename:
            payload[angle] = f.read()
        else:
            payload[angle] = None

    result = analyze_face_set(payload)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
