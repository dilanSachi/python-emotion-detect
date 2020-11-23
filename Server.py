from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
from Predictor import loadModel, predictEmotion

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/', methods=['POST'])
def test():
    r = request

    nparr = np.frombuffer(r.data, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imwrite("temp.png", img)

    emotion = predictEmotion("temp.png")
    response = {'message': emotion}
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

loadModel()
# start flask app
app.run(host="0.0.0.0", port=5000)