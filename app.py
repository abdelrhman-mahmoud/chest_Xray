from flask import Flask, request, render_template, jsonify
import torch 
import os 
from vit import run_inference as vit_run_infrence, predict_single_image as vit_predict
from inception import run_inference as inception_run_inference, predict_single_image as inception_predict
from download_models import download_extract_model
download_extract_model()
app = Flask(__name__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
vit_model, vit_processor, class_names = vit_run_infrence()
inception_model, transforms, class_names = inception_run_inference()
inception_model = inception_model.to(device)
vit_model = vit_model.to(device)

UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/vit', methods = ['POST'])
def predict_vit():
    if 'image' not in request.files:
        return jsonify({'error': 'no image selcted'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try :
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        prediction, confidence = vit_predict(image_path, vit_model, vit_processor, class_names, device)
        # os.remove(image_path)

        return render_template('index.html', 
                        prediction=prediction, 
                        confidence=f"{confidence:.4f}",
                        model='vit'.upper())

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/predict/inception', methods=['POST'])
def predict_inception():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        prediction, confidence = inception_predict(
            image_path, inception_model, transforms, class_names, device
        )
        
        # os.remove(image_path)
        
        return render_template('index.html', 
                                prediction=prediction, 
                                confidence=f"{confidence:.4f}",
                                model='inception'.upper())
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True, host= '0.0.0.0')

        
