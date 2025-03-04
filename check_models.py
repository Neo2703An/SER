import os
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_models():
    """Check if models exist and display their information."""
    models_dir = 'models'
    model_files = ['cnn_model.h5', 'lstm_model.h5', 'hybrid_model.h5']
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory '{models_dir}' does not exist.")
        return
    
    logger.info(f"Checking models in '{models_dir}'...")
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        if os.path.exists(model_path):
            logger.info(f"Model '{model_file}' exists.")
            
            # Get file size
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            logger.info(f"  - Size: {size_mb:.2f} MB")
            
            # Load model and get summary
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"  - Successfully loaded model")
                logger.info(f"  - Input shape: {model.input_shape}")
                logger.info(f"  - Output shape: {model.output_shape}")
                
                # Count parameters
                trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
                logger.info(f"  - Trainable parameters: {trainable_params:,}")
                logger.info(f"  - Non-trainable parameters: {non_trainable_params:,}")
                logger.info(f"  - Total parameters: {trainable_params + non_trainable_params:,}")
            except Exception as e:
                logger.error(f"  - Error loading model: {e}")
        else:
            logger.warning(f"Model '{model_file}' does not exist yet.")
    
    # Check for checkpoint files
    checkpoint_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') and not f in model_files]
    if checkpoint_files:
        logger.info(f"Found {len(checkpoint_files)} checkpoint files:")
        for checkpoint in checkpoint_files:
            logger.info(f"  - {checkpoint}")

if __name__ == "__main__":
    check_models() 