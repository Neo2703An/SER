import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten, Input, TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Activation, GRU, Bidirectional, LeakyReLU, Add, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import logging
import random
from scipy.signal import butter, lfilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = "data"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define emotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function for data augmentation
def augment_audio(y, sr, augmentation_type=None):
    """Apply various augmentation techniques to audio data."""
    if augmentation_type is None:
        augmentation_type = random.choice(['pitch', 'speed', 'noise', 'shift', 'none'])
    
    if augmentation_type == 'pitch':
        # Pitch shift (up or down by 0-2 semitones)
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    elif augmentation_type == 'speed':
        # Time stretching (speed up or slow down by 0.8-1.2 factor)
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y, rate=rate)
    
    elif augmentation_type == 'noise':
        # Add random noise
        noise_factor = random.uniform(0.005, 0.015)
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    
    elif augmentation_type == 'shift':
        # Time shift (shift by -0.5 to 0.5 seconds)
        shift = int(random.uniform(-0.5, 0.5) * sr)
        if shift > 0:
            return np.pad(y, (shift, 0), mode='constant')[0:len(y)]
        else:
            return np.pad(y, (0, -shift), mode='constant')[0:len(y)]
    
    else:  # 'none'
        return y

# Function to apply bandpass filter
def bandpass_filter(data, lowcut=300, highcut=8000, fs=22050, order=5):
    """Apply bandpass filter to focus on speech frequencies."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Function to extract features from audio files
def extract_features(file_path, feature_type='mfcc', augment=False):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Apply bandpass filter to focus on speech frequencies
        y = bandpass_filter(y)
        
        # Apply data augmentation if requested
        if augment:
            y = augment_audio(y, sr)
        
        # If audio is too short, pad it
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), 'constant')
        
        # If audio is too long, truncate it
        if len(y) > sr * 5:  # Limit to 5 seconds
            y = y[:sr * 5]
        
        # Extract features based on feature type
        if feature_type == 'mfcc':
            # Extract MFCCs with more coefficients
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Add delta and delta-delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            combined_mfccs = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
            
            # Resize to expected input shape
            if combined_mfccs.shape[1] < 128:
                combined_mfccs = np.pad(combined_mfccs, ((0, 0), (0, 128 - combined_mfccs.shape[1])), 'constant')
            else:
                combined_mfccs = combined_mfccs[:, :128]
            
            return combined_mfccs
            
        elif feature_type == 'mel':
            # Extract mel spectrogram with more mel bands
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to expected input shape
            if mel_spec_db.shape[1] < 128:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])), 'constant')
            else:
                mel_spec_db = mel_spec_db[:, :128]
            
            return mel_spec_db
        
        elif feature_type == 'combined':
            # Extract multiple feature types
            # 1. MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 2. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 3. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 4. Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff = rolloff.reshape(1, -1)
            
            # 5. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr = zcr.reshape(1, -1)
            
            # Combine all features
            combined = np.concatenate([mfccs, contrast, chroma, rolloff, zcr])
            
            # Resize to expected input shape
            if combined.shape[1] < 128:
                combined = np.pad(combined, ((0, 0), (0, 128 - combined.shape[1])), 'constant')
            else:
                combined = combined[:, :128]
            
            return combined
    
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None

# Function to load dataset
def load_dataset(augment_data=True):
    features = []
    labels = []
    
    # Process each dataset
    datasets = ['ravdess', 'crema', 'tess', 'savee']
    total_files = 0
    processed_files = 0
    
    # First count total files
    for dataset in datasets:
        dataset_path = os.path.join(DATA_PATH, dataset)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path {dataset_path} does not exist. Skipping.")
            continue
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    total_files += 1
    
    logger.info(f"Found {total_files} audio files across all datasets")
    
    # Now process each dataset
    for dataset in datasets:
        dataset_path = os.path.join(DATA_PATH, dataset)
        
        if not os.path.exists(dataset_path):
            continue
        
        logger.info(f"Processing dataset: {dataset}")
        dataset_files = 0
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    processed_files += 1
                    dataset_files += 1
                    
                    if processed_files % 100 == 0:
                        logger.info(f"Processed {processed_files}/{total_files} files ({(processed_files/total_files)*100:.1f}%)")
                    
                    # Extract emotion from filename or directory structure
                    emotion = None
                    
                    # RAVDESS format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                    if dataset == 'ravdess':
                        emotion_code = int(file.split('-')[2])
                        emotion_map = {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 
                                      5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
                        emotion = emotion_map.get(emotion_code)
                    
                    # CREMA-D format: ActorID_Sentence_Emotion_Intensity.wav
                    elif dataset == 'crema':
                        emotion_code = file.split('_')[2]
                        emotion_map = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 
                                      'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'}
                        emotion = emotion_map.get(emotion_code)
                    
                    # TESS format: OAF_emotion_word.wav or YAF_emotion_word.wav
                    elif dataset == 'tess':
                        emotion = file.split('_')[1].lower()
                        if emotion == 'ps':
                            emotion = 'surprise'
                    
                    # SAVEE format: emotion_statement_repetition.wav
                    elif dataset == 'savee':
                        # First check for two-letter emotion codes
                        if file.startswith('sa') and len(file) > 2 and file[2] in ['0', '1', '_']:
                            emotion = 'sad'
                        elif file.startswith('su') and len(file) > 2 and file[2] in ['0', '1', '_']:
                            emotion = 'surprise'
                        else:
                            # Then check for single-letter emotion codes
                            emotion_code = file[0]
                            emotion_map = {'n': 'neutral', 'h': 'happy', 
                                          'a': 'angry', 'f': 'fear', 'd': 'disgust'}
                            emotion = emotion_map.get(emotion_code)
                    
                    # Skip if emotion not identified
                    if not emotion or emotion not in EMOTIONS:
                        logger.warning(f"Could not identify emotion for {file_path}. Skipping.")
                        continue
                    
                    # Extract features
                    mfcc_features = extract_features(file_path, 'mfcc')
                    mel_features = extract_features(file_path, 'mel')
                    combined_features = extract_features(file_path, 'combined')
                    
                    if mfcc_features is not None and mel_features is not None and combined_features is not None:
                        features.append({
                            'mfcc': mfcc_features,
                            'mel': mel_features,
                            'combined': combined_features
                        })
                        labels.append(emotion)
                        
                        # Add augmented versions if requested
                        if augment_data:
                            # Add 1-2 augmented versions for underrepresented emotions
                            if emotion in ['disgust', 'fear', 'surprise']:
                                num_augmentations = 2
                            else:
                                num_augmentations = 1
                                
                            for _ in range(num_augmentations):
                                mfcc_aug = extract_features(file_path, 'mfcc', augment=True)
                                mel_aug = extract_features(file_path, 'mel', augment=True)
                                combined_aug = extract_features(file_path, 'combined', augment=True)
                                
                                if mfcc_aug is not None and mel_aug is not None and combined_aug is not None:
                                    features.append({
                                        'mfcc': mfcc_aug,
                                        'mel': mel_aug,
                                        'combined': combined_aug
                                    })
                                    labels.append(emotion)
        
        logger.info(f"Finished processing {dataset} dataset: {dataset_files} files")
    
    logger.info(f"Total processed files: {len(features)}")
    # Count emotions
    emotion_counts = {}
    for emotion in labels:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    
    for emotion, count in emotion_counts.items():
        logger.info(f"Emotion '{emotion}': {count} samples")
    
    return features, labels

# Function to build improved CNN model
def build_cnn_model(input_shape=(128, 128, 1), num_classes=7):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        
        # Dense layers
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(128, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with a lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to build improved LSTM model
def build_lstm_model(input_shape=(128, 60), num_classes=7):
    model = Sequential([
        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        
        # Dense layers
        Dense(128, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with a lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to build improved Hybrid CNN-LSTM model
def build_hybrid_model(input_shape=(128, 128, 1), num_classes=7):
    # CNN part
    cnn_input = Input(shape=input_shape)
    
    # First convolutional block with residual connection
    x = Conv2D(32, (3, 3), padding='same')(cnn_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Second convolutional block with residual connection
    x_res = Conv2D(64, (1, 1), strides=(2, 2))(cnn_input)  # Shortcut connection
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])  # Add residual connection
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Reshape for LSTM
    x = tf.keras.layers.Reshape((-1, 64 * (input_shape[0] // 4)))(x)
    
    # LSTM part
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    
    # Dense layers
    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=cnn_input, outputs=outputs)
    
    # Compile model with a lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train models
def train_models():
    # Load dataset
    logger.info("Loading dataset with augmentation...")
    features, labels = load_dataset(augment_data=True)
    
    if not features:
        logger.error("No features extracted. Exiting.")
        return
    
    logger.info(f"Dataset loaded with {len(features)} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = to_categorical(y_encoded)
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Split dataset
    X_mfcc = np.array([f['mfcc'] for f in features])
    X_mel = np.array([f['mel'] for f in features])
    X_combined = np.array([f['combined'] for f in features])
    
    # Normalize features
    X_mfcc_reshaped = X_mfcc.reshape(X_mfcc.shape[0], -1)
    X_mel_reshaped = X_mel.reshape(X_mel.shape[0], -1)
    X_combined_reshaped = X_combined.reshape(X_combined.shape[0], -1)
    
    scaler_mfcc = StandardScaler()
    scaler_mel = StandardScaler()
    scaler_combined = StandardScaler()
    
    X_mfcc_scaled = scaler_mfcc.fit_transform(X_mfcc_reshaped)
    X_mel_scaled = scaler_mel.fit_transform(X_mel_reshaped)
    X_combined_scaled = scaler_combined.fit_transform(X_combined_reshaped)
    
    X_mfcc = X_mfcc_scaled.reshape(X_mfcc.shape)
    X_mel = X_mel_scaled.reshape(X_mel.shape)
    X_combined = X_combined_scaled.reshape(X_combined.shape)
    
    # Reshape for different models
    X_mfcc_lstm = X_mfcc.transpose(0, 2, 1)  # (samples, time_steps, features)
    X_mel_cnn = X_mel.reshape(X_mel.shape[0], X_mel.shape[1], X_mel.shape[2], 1)  # (samples, height, width, channels)
    X_combined_lstm = X_combined.transpose(0, 2, 1)  # (samples, time_steps, features)
    
    # Split into train and test sets
    X_mfcc_lstm_train, X_mfcc_lstm_test, X_mel_cnn_train, X_mel_cnn_test, X_combined_lstm_train, X_combined_lstm_test, y_train, y_test = train_test_split(
        X_mfcc_lstm, X_mel_cnn, X_combined_lstm, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        ModelCheckpoint(filepath=os.path.join(SAVE_DIR, '{epoch:02d}-{val_accuracy:.4f}.h5'), 
                        monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train CNN model
    logger.info("Training CNN model...")
    cnn_model = build_cnn_model(input_shape=(128, 128, 1), num_classes=len(EMOTIONS))
    logger.info(f"CNN model summary: {cnn_model.summary()}")
    cnn_history = cnn_model.fit(
        X_mel_cnn_train, y_train,
        validation_data=(X_mel_cnn_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    cnn_model.save(os.path.join(SAVE_DIR, 'cnn_model.h5'))
    logger.info("CNN model training completed and saved")
    
    # Train LSTM model
    logger.info("Training LSTM model...")
    lstm_model = build_lstm_model(input_shape=(128, X_combined_lstm.shape[2]), num_classes=len(EMOTIONS))
    logger.info(f"LSTM model summary: {lstm_model.summary()}")
    lstm_history = lstm_model.fit(
        X_combined_lstm_train, y_train,
        validation_data=(X_combined_lstm_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    lstm_model.save(os.path.join(SAVE_DIR, 'lstm_model.h5'))
    logger.info("LSTM model training completed and saved")
    
    # Train Hybrid CNN-LSTM model
    logger.info("Training Hybrid CNN-LSTM model...")
    hybrid_model = build_hybrid_model(input_shape=(128, 128, 1), num_classes=len(EMOTIONS))
    logger.info(f"Hybrid model summary: {hybrid_model.summary()}")
    hybrid_history = hybrid_model.fit(
        X_mel_cnn_train, y_train,
        validation_data=(X_mel_cnn_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    hybrid_model.save(os.path.join(SAVE_DIR, 'hybrid_model.h5'))
    logger.info("Hybrid model training completed and saved")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['accuracy'], label='CNN Training')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation')
    plt.plot(lstm_history.history['accuracy'], label='LSTM Training')
    plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation')
    plt.plot(hybrid_history.history['accuracy'], label='Hybrid Training')
    plt.plot(hybrid_history.history['val_accuracy'], label='Hybrid Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['loss'], label='CNN Training')
    plt.plot(cnn_history.history['val_loss'], label='CNN Validation')
    plt.plot(lstm_history.history['loss'], label='LSTM Training')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')
    plt.plot(hybrid_history.history['loss'], label='Hybrid Training')
    plt.plot(hybrid_history.history['val_loss'], label='Hybrid Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
    
    logger.info("Training completed. Models saved to 'models' directory.")

if __name__ == "__main__":
    train_models() 