import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import logging
import hashlib
import base64
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define emotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cache for storing analysis results
analysis_cache = {}

# Load models
def load_models():
    models = {}
    try:
        # CNN Model
        models['cnn'] = tf.keras.models.load_model('models/cnn_model.h5')
        logger.info("CNN model loaded successfully")
        
        # LSTM Model
        models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5')
        logger.info("LSTM model loaded successfully")
        
        # Hybrid CNN-LSTM Model
        models['hybrid'] = tf.keras.models.load_model('models/hybrid_model.h5')
        logger.info("Hybrid model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Create dummy models for testing if real models are not available
        models = create_dummy_models()
    
    return models

def create_dummy_models():
    logger.warning("Creating dummy models for testing")
    models = {}
    
    # Create a simple dummy model structure for each type
    # CNN model
    cnn_input = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    cnn_output = tf.keras.layers.Dense(len(EMOTIONS), activation='softmax')(x)
    models['cnn'] = tf.keras.Model(inputs=cnn_input, outputs=cnn_output)
    
    # LSTM model
    lstm_input = tf.keras.Input(shape=(128, 13))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(lstm_input)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    lstm_output = tf.keras.layers.Dense(len(EMOTIONS), activation='softmax')(x)
    models['lstm'] = tf.keras.Model(inputs=lstm_input, outputs=lstm_output)
    
    # Hybrid CNN-LSTM model
    hybrid_input = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(hybrid_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    hybrid_output = tf.keras.layers.Dense(len(EMOTIONS), activation='softmax')(x)
    models['hybrid'] = tf.keras.Model(inputs=hybrid_input, outputs=hybrid_output)
    
    return models

# Feature extraction
def extract_features(file_path, model_type):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # If audio is too short, pad it
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), 'constant')
        
        # If audio is too long, truncate it
        if len(y) > sr * 5:  # Limit to 5 seconds
            y = y[:sr * 5]
        
        # Extract features based on model type
        if model_type == 'cnn' or model_type == 'hybrid':
            # Extract mel spectrogram for CNN and Hybrid models
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to expected input shape
            if mel_spec_db.shape[1] < 128:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])), 'constant')
            else:
                mel_spec_db = mel_spec_db[:, :128]
            
            # Reshape for CNN input (samples, height, width, channels)
            features = mel_spec_db.reshape(1, 128, 128, 1)
            
        elif model_type == 'lstm':
            # Extract MFCCs for LSTM
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Resize to expected input shape
            if mfccs.shape[1] < 128:
                mfccs = np.pad(mfccs, ((0, 0), (0, 128 - mfccs.shape[1])), 'constant')
            else:
                mfccs = mfccs[:, :128]
            
            # Reshape for LSTM input (samples, time steps, features)
            features = mfccs.T.reshape(1, 128, 13)
        
        return features, y, sr
    
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        # Return dummy features if extraction fails
        if model_type == 'cnn' or model_type == 'hybrid':
            return np.zeros((1, 128, 128, 1)), None, None
        else:
            return np.zeros((1, 128, 13)), None, None

# Extract audio visualizations
def extract_audio_visualizations(y, sr):
    try:
        visualizations = {}
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Generate MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Generate chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Generate spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Create plots
        plt.switch_backend('Agg')
        
        # Mel Spectrogram plot
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
        ax.set_title('Mel Spectrogram')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        canvas = FigureCanvas(fig)
        img_buffer = io.BytesIO()
        canvas.print_png(img_buffer)
        img_buffer.seek(0)
        visualizations['mel_spectrogram'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # MFCC plot
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        ax.set_title('MFCC')
        fig.colorbar(img, ax=ax)
        canvas = FigureCanvas(fig)
        img_buffer = io.BytesIO()
        canvas.print_png(img_buffer)
        img_buffer.seek(0)
        visualizations['mfcc'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Waveform plot
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#4776E6')
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        canvas = FigureCanvas(fig)
        img_buffer = io.BytesIO()
        canvas.print_png(img_buffer)
        img_buffer.seek(0)
        visualizations['waveform'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return visualizations
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return {}

# Predict emotion
def predict_emotion(audio_features, model, model_type):
    try:
        # Make prediction
        predictions = model.predict(audio_features)
        
        # Get the index of the highest probability
        predicted_index = np.argmax(predictions[0])
        
        # Get the corresponding emotion and confidence
        emotion = EMOTIONS[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'model': model_type
        }
    
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        # Return a default prediction if prediction fails
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'model': model_type
        }

# Deterministic mock prediction for testing without models
def mock_predict(model_type, file_hash):
    # Use the file hash to create a deterministic prediction
    hash_value = int(file_hash, 16)
    
    # Deterministic emotion selection based on hash and model type
    if model_type == 'cnn':
        emotion_index = hash_value % len(EMOTIONS)
    elif model_type == 'lstm':
        emotion_index = (hash_value + 2) % len(EMOTIONS)
    else:  # hybrid
        emotion_index = (hash_value + 4) % len(EMOTIONS)
    
    emotion = EMOTIONS[emotion_index]
    
    # Deterministic confidence based on hash
    confidence = 0.6 + ((hash_value % 35) / 100)  # Between 0.6 and 0.95
    
    return {
        'emotion': emotion,
        'confidence': float(confidence),
        'model': model_type
    }

# Calculate file hash for caching
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Load models
models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    model_type = request.form.get('model', 'cnn')  # Default to CNN if not specified
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Calculate file hash for caching
            file_hash = calculate_file_hash(file_path)
            cache_key = f"{file_hash}_{model_type}"
            
            # Check if we have cached results
            if cache_key in analysis_cache:
                logger.info(f"Using cached result for {filename} with model {model_type}")
                result = analysis_cache[cache_key]
            else:
                # Check if we should use real models or mock predictions
                use_real_models = os.path.exists(f'models/{model_type}_model.h5')
                
                if use_real_models:
                    # Extract features
                    features, audio_data, sample_rate = extract_features(file_path, model_type)
                    
                    # Predict emotion
                    result = predict_emotion(features, models[model_type], model_type)
                    
                    # Add audio visualizations if audio data is available
                    if audio_data is not None and sample_rate is not None:
                        visualizations = extract_audio_visualizations(audio_data, sample_rate)
                        result['visualizations'] = visualizations
                else:
                    # Use deterministic mock prediction for testing
                    result = mock_predict(model_type, file_hash)
                    
                    # Load audio for visualizations even in mock mode
                    try:
                        audio_data, sample_rate = librosa.load(file_path, sr=22050)
                        visualizations = extract_audio_visualizations(audio_data, sample_rate)
                        result['visualizations'] = visualizations
                    except Exception as e:
                        logger.error(f"Error loading audio for visualizations: {e}")
                
                # Cache the result
                analysis_cache[cache_key] = result
                logger.info(f"Cached result for {filename} with model {model_type}")
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True) 