from ultralytics import YOLO
from flask import request, Response, Flask
from PIL import Image
import json
import os
import tempfile
from roboflow import Roboflow
from waitress import serve

app = Flask(__name__)

# Initialize Roboflow
rf = Roboflow(api_key="kcPS04kFy2hy9jSexUE4")
project = rf.workspace("keypointsdetectionoko").project("keypoints_annotation")
model = project.version(1).model

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("app/templates/index.html") as file:
        return file.read()


@app.route("/charts")
def chart():
    """
    Handler for serving the chart.html file.
    :return: Content of chart.html file
    """
    with open("app/templates/chart.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file",
    passes it through YOLOv8 object detection
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding
    boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    predictions = detect_objects_on_image(Image.open(buf.stream))
    return Response(
        json.dumps(predictions),
        mimetype='application/json'
    )


def detect_objects_on_image(buf):
    """
    Function receives an image,
    saves it to a temporary file in the current working directory,
    passes the file path through the Roboflow model for object detection,
    and returns the predicted bounding boxes.
    :param buf: Input image file stream
    :return: Predicted bounding boxes in JSON format
    """
    # Save the image buffer to a temporary file in the current working directory
    with tempfile.NamedTemporaryFile(suffix='.jpg', dir='.', delete=False) as temp_file:
        temp_file_path = temp_file.name
        buf.save(temp_file_path)

    # Predict using the Roboflow model
    predictions = model.predict(temp_file_path).json()

    # Remove the temporary file
    os.unlink(temp_file_path)
    
    # Process the predictions
    output = []
    for prediction in predictions["predictions"]:
        for pred in prediction["predictions"]:
            x1 = pred["x"]
            y1 = pred["y"]
            x2 = x1 + pred["width"]
            y2 = y1 + pred["height"]
            label = pred["class"]
            confidence = pred["confidence"]
            width = pred["width"]
            height = pred["height"]
            keypoints = pred["keypoints"]  # You can modify this part to include keypoints if needed
            bbox = {
                "x1": x1 - (width / 2),
                "y1": y1 - (height / 2),
                "x2": x2 - (width / 2),
                "y2": y2 - (height / 2),
                "label": label,
                "confidence": confidence,
                "width": width,
                "height": height,
                "keypoints": keypoints,
            }
            output.append(bbox)

    return output


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)