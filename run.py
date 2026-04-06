

def extract_prosodic_features(y, sr):
    # Pitch (F0) statistics
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Energy/Intensity
    energy = np.sum(y**2) / len(y)
    
    # Speaking rate (using zero-crossing rate as a proxy)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0].mean()
    
    return np.array([pitch_mean, pitch_std, energy, zero_crossing_rate])

def load_crema_d_data(CREMA_D_PATH):
    data = []
    for filename in os.listdir(CREMA_D_PATH):
        if filename.endswith(".wav"):
            file_path = os.path.join(CREMA_D_PATH, filename)
            
            # Extract information from filename
            parts = filename.split('_')
            if len(parts) < 4:
                print(f"Skipping file with unexpected format: {filename}")
                continue
            
            emotion = parts[2]
            intensity = parts[3].split('.')[0]  # Remove .wav extension
            
            # Load audio file and extract features
            y, sr = librosa.load(file_path, duration=3)  # Load up to 3 seconds
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs.T, axis=0)
            
            # Extract prosodic features
            prosodic_features = extract_prosodic_features(y, sr)
            
            # Combine MFCC and prosodic features
            combined_features = np.concatenate([mfcc_mean, prosodic_features])
            
            # Append data
            data.append({
                'filename': filename,
                'emotion': emotion,
                'intensity': intensity,
                'features': combined_features
            })
    
    df = pd.DataFrame(data)
    print(f"Unique intensity values: {df['intensity'].unique()}")
    return df

def calculate_stress_level(emotion, intensity):
    primary_stress = {'ANG': 0.8, 'FEA': 0.8, 'SAD': 0.7}
    secondary_stress = {'DIS': 0.6}
    
    base_stress = 0.5  # Neutral baseline
    
    if emotion in primary_stress:
        stress_factor = primary_stress[emotion]
    elif emotion in secondary_stress:
        stress_factor = secondary_stress[emotion]
    else:  # Neutral
        return base_stress
    
    intensity_mapping = {'LO': 0.5, 'MD': 0.75, 'HI': 1.0, 'XX': 0.75}  # XX for Unspecified
    intensity_factor = intensity_mapping.get(intensity, 0.75)  # Default to 0.75 if not found
    
    stress_level = base_stress + (stress_factor * intensity_factor)
    return max(0, min(stress_level, 1))  # Ensure stress level is between 0 and 1

def load_data(df):
    df['stress_level'] = df.apply(lambda row: calculate_stress_level(row['emotion'], row['intensity']), axis=1)
    
    # Prepare features and labels
    X = np.stack(df['features'].values)
    y = df['stress_level'].values
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    return X, y

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

CREMA_D_PATH = "./CREMA-D/AudioWAV"  # Update this path

# [Function definitions for extract_prosodic_features, load_crema_d_data, calculate_stress_level, and load_data remain the same]

def stress_detection_model(CREMA_D_PATH):
    # Load and prepare data
    df = load_crema_d_data(CREMA_D_PATH)
    X, y = load_data(df)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train CatBoost Regressor
    catboost_model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=0)
    catboost_model.fit(X_train_scaled, y_train)

    # Get CatBoost predictions for the test set
    y_pred = catboost_model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Results:")
    print(f" MAE: {mae:.4f}")
    print(f" MSE: {mse:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f" R-squared: {r2:.4f}")

    # Calculate threshold-based accuracy
    threshold = 0.35
    within_threshold = np.abs(y_test - y_pred) < threshold
    accuracy = np.mean(within_threshold)

    print(f"\nThreshold-based Accuracy:")
    print(f" Accuracy (within {threshold:.2f} threshold): {accuracy:.4f}")

if __name__ == "__main__":
    stress_detection_model(CREMA_D_PATH)