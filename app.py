from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Render with Python 3.10.14 🎉"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
