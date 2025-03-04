# Speech Emotion Recognition System

This application uses deep learning models to analyze the emotional content of speech audio files. It provides three different models for analysis: CNN, LSTM, and Hybrid CNN-LSTM.

## Features

- Upload and analyze audio files (WAV, MP3, etc.)
- Choose between three different deep learning models:
  - CNN (Convolutional Neural Network): Best for spectral patterns in audio
  - LSTM (Long Short-Term Memory): Best for temporal patterns in audio
  - Hybrid CNN-LSTM: Combines the strengths of both models
- Real-time audio playback
- Visual display of emotion detection results with confidence scores

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Make sure you have the data folder with the following datasets:
   - RAVDESS
   - CREMA-D
   - TESS
   - SAVEE

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`

3. Select a model (CNN, LSTM, or Hybrid)

4. Upload an audio file by clicking on the upload area or dragging and dropping a file

5. Click "Analyze Emotion" to process the audio

6. View the results showing the detected emotion and confidence level

7. Click "Analyze New Sample" to upload and analyze another audio file

## Model Training

If you want to train the models yourself:

1. Make sure you have the data folder with the required datasets

2. Run the training script:
   ```
   python train_models.py
   ```

This will train all three models (CNN, LSTM, and Hybrid CNN-LSTM) and save them to the `models` folder.

## Project Structure

- `app.py`: Main Flask application
- `train_models.py`: Script to train the models
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files
- `models/`: Trained model files
- `data/`: Speech emotion datasets
- `uploads/`: Temporary folder for uploaded audio files

## Technologies Used

- Python
- TensorFlow/Keras
- Flask
- Librosa (audio processing)
- HTML/CSS/JavaScript

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RAVDESS, CREMA-D, TESS, and SAVEE datasets for providing the training data
- TensorFlow and Keras for the deep learning framework
- Librosa for audio processing capabilities 