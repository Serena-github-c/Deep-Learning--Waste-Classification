from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
from waitress import serve
from flask import send_from_directory


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model('xception_299_04_0.934.keras')

# Categories and biodegradability mapping
categories = ["ewaste", "food_waste", "leaf_waste", "metal_cans", "paper_waste", 
             "plastic_bags", "plastic_bottles", "wood_waste"]

subcategory_to_biodegradable = {
    0: 'non_biodegradable',
    1: 'biodegradable',
    2: 'biodegradable',
    3: 'non_biodegradable',
    4: 'biodegradable',
    5: 'non_biodegradable',
    6: 'non_biodegradable',
    7: 'biodegradable'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def make_prediction(file_path):
    img = load_img(file_path, target_size=(299, 299))
    x = np.array(img)
    X = preprocess_input(np.array([x]))
    pred = model.predict(X)
    predicted_index = np.argmax(pred[0])
    biodegradability = subcategory_to_biodegradable[predicted_index]
    
    # Include confidence scores for all categories
    confidence_scores = dict(zip(categories, pred[0].tolist()))
    
    final_pred = {
        'predicted_category': categories[predicted_index],
        'biodegradability': biodegradability,
        'confidence_scores': confidence_scores
    }
    return final_pred



@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = make_prediction(filepath)
        
        # Clean up - remove the uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a route to serve the HTML file
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')
    
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Development : flask
    # app.run(debug=True)
    
    # Production : waitress
    serve(app, host='0.0.0.0', port=9696, threads=4)