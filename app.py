from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model
model = load_model('./model/my_model.h5')

def prepare_image(image, target):
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image to the target size
    image = image.resize(target)
    # Convert the image to an array
    image = img_to_array(image)
    # Expand the dimensions to match the model's input shape
    image = np.expand_dims(image, axis=0)
    # Normalize the image to [0, 1] range
    image = image / 255.0
    return image

@app.route("/", methods=["GET"])
def index():
    # Render the index.html page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the file from the request
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Read the image bytes
        img_bytes = file.read()
        # Open the image
        image = Image.open(io.BytesIO(img_bytes))
        # Preprocess the image
        processed_image = prepare_image(image, target=(256, 256))  # Adjust the target size if needed

        # Make a prediction
        prediction = model.predict(processed_image).tolist()
        
        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction})
    
    return jsonify({"error": "Invalid request method"}), 405

if __name__ == "__main__":
    app.run(debug=True)
