from flask import Flask, request, jsonify
from load_model import *
from crop_yolo import *
import os
import cv2
import base64

app = Flask(__name__)

@app.route("/predict", methods=["POST"])


def inference():
    image_data = request.form.get("image_data")
    image = preprocessing_byte(image_data)
    image_np = np.array(image)
    crop_img = estimate_blur(image_np)

    results = []  # Initialize the list to store results

    for i in range(len(crop_img)):
        result = predict_function(crop_img[i])
        results.append(result)
        save_image_to_pic_folder(crop_img[i],i)

    # Call query_openai with both results and images
    final_result = query_openai(results)

    return final_result


if __name__ == "__main__":
    app.run(debug=True,port=5002)