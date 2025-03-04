import requests
import os
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_models():
    """Test the trained models by making requests to the Flask application."""
    # URL of the Flask application
    url = 'http://127.0.0.1:5000/analyze'
    
    # Test audio file path (use a sample from one of the datasets)
    test_files = [
        'data/ravdess/Actor_01/03-01-03-01-01-01-01.wav',  # Neutral
        'data/ravdess/Actor_01/03-01-05-01-01-01-01.wav',  # Angry
        'data/ravdess/Actor_01/03-01-06-01-01-01-01.wav',  # Fear
        'data/ravdess/Actor_01/03-01-04-01-01-01-01.wav'   # Happy
    ]
    
    # Models to test
    models = ['cnn', 'lstm', 'hybrid']
    
    # Test each model with each file
    for model in models:
        logger.info(f"Testing {model.upper()} model...")
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                logger.warning(f"Test file {test_file} does not exist. Skipping.")
                continue
            
            # Get emotion from filename (for RAVDESS)
            emotion_code = int(os.path.basename(test_file).split('-')[2])
            emotion_map = {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 
                          5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
            expected_emotion = emotion_map.get(emotion_code, 'unknown')
            
            logger.info(f"  Testing with file: {test_file} (Expected: {expected_emotion})")
            
            # Create form data
            files = {'audio': open(test_file, 'rb')}
            data = {'model': model}
            
            try:
                # Make request
                response = requests.post(url, files=files, data=data)
                
                # Check response
                if response.status_code == 200:
                    result = response.json()
                    predicted_emotion = result.get('emotion', 'unknown')
                    confidence = result.get('confidence', 0.0)
                    
                    logger.info(f"    Predicted: {predicted_emotion} (Confidence: {confidence:.2f})")
                    logger.info(f"    Match: {'✓' if predicted_emotion == expected_emotion else '✗'}")
                    
                    # Check if visualizations are included
                    if 'visualizations' in result:
                        logger.info(f"    Visualizations included: {list(result['visualizations'].keys())}")
                else:
                    logger.error(f"    Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                logger.error(f"    Error making request: {e}")
            
            # Close file
            files['audio'].close()
            
            # Wait a bit between requests
            time.sleep(1)
        
        logger.info("")

if __name__ == "__main__":
    # Wait for the Flask application to start
    logger.info("Waiting for Flask application to start...")
    time.sleep(5)
    
    # Test the models
    test_models() 