from flask import Flask
app = Flask(__name__)

@app.route('/get-measurements')
def get_measurements():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()