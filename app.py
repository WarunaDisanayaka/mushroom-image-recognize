from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

# Load our saved model (Mushroom classification model)
model_path = "Mushroom Classification Model.h5"
model = None

def load_mushroom_model():
    global model
    model = load_model(model_path)

# Load the model outside of the Flask app
load_mushroom_model()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# initialize the flask
app = Flask(__name__)
CORS(app)

# Set the static folder
app.static_folder = 'static'

# routing 
@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/Mushroom-classification-predict")
def predict_page():
    return render_template("input.html")
    
@app.route("/Mushroom-classification-predict", methods=["POST"])
def predict():
    try:
        # Get the image file
        imageFile = request.files.get("image_file")
        
        if imageFile and allowed_file(imageFile.filename):
            # Read the image file as a PIL image
            image_bytes = imageFile.read()
            inputImage = Image.open(io.BytesIO(image_bytes))
            inputImage = inputImage.convert("RGB")  # Ensure it's RGB
            inputImage = inputImage.resize((224, 224))  # Resize to match model input
            
            # Preprocess the image
            inputImage = image.img_to_array(inputImage)
            inputImage = np.expand_dims(inputImage, axis=0)
            inputImage = inputImage / 255.0
            
            # Extract features using the pre-trained model
            features = model.predict(inputImage)
            
            # Ensure the output is a 2D array with probabilities for each class
            if len(features.shape) == 2:
                features = features[0]  # Flatten the output if necessary
            else:
                raise ValueError("Unexpected shape of prediction output.")
            
            # Define class names (ensure they match the number of classes in the model)
            class_names = ['Not Mushroom', 'Mushroom']  # Update if necessary
            
            # Find the predicted class index
            predicted_class_index = np.argmax(features)
            confidence = features[predicted_class_index]
            
            # Set a confidence threshold (e.g., 0.90)
            threshold = 0.90
            if confidence < threshold:
                prediction = "Not Mushroom"
            else:
                predicted_class = class_names[predicted_class_index]
                prediction = f"Predicted: {predicted_class}"
            
            return jsonify({"prediction": prediction})
        else:
            return jsonify({"prediction": "Invalid file type"})
    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"})

# run the application
if __name__ == "__main__":
    app.run(debug=True)
