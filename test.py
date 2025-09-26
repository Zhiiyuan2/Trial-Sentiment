from flask import Flask, render_template, request
from pathlib import Path

# Point Flask explicitly to the templates folder next to this file
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "")
    return render_template("index.html", message=message)

if __name__ == "__main__":
    # Run: python app.py  (then open http://127.0.0.1:5000)
    app.run(debug=True)
 # Run: python app.py  (then open http://127.0.0.1:5000)
