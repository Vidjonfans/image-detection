from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Flask App Running on Render (Python 3.13)"
