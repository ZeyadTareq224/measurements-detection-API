from flask import Flask
from .model import MeasurementDetector

app = Flask(__name__)

@app.route('/get-measurements')
def get_measurements():
    obj = MeasurementDetector('test_img.jpeg')
    return obj.calculate_real_measurements()

if __name__ == '__main__':
    app.run()