import os
import pickle
import openai
import librosa
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


# 📌 Load Training Data (RAVDESS Dataset)
def load_training_data():
    """Loads and extracts features from the RAVDESS dataset."""
    data_dir = "ravdess_data"  # Ensure this folder contains WAV files

    features, labels = [], []

    for file in os.listdir(data_dir):
        if file.endswith(".wav"):  
            file_path = os.path.join(data_dir, file)
            audio, sr = librosa.load(file_path, sr=44100)

            # Extract features
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

            # Combine features
            feature_vector = np.hstack([mfccs, chroma, mel])
            features.append(feature_vector)

            # Extract label (modify this based on dataset filename structure)
            emotion_label = int(file.split("-")[2])  
            labels.append(emotion_label)

    features, labels = np.array(features), np.array(labels)

    # Encode labels into numbers
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    print(f"✅ Loaded {len(features)} emotion samples from RAVDESS.")
    return features, labels, encoder

# 📌 Train the Model
def train_model():
    """Loads data, trains model, and saves it."""
    X_train, y_train, encoder = load_training_data()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train with optimized parameters
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", 
                          learning_rate="adaptive", max_iter=1000)
    model.fit(X_scaled, y_train)

    # Save trained model and encoder
    with open("trained_emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("✅ Model trained and saved successfully!")

# Train model if not already trained
if not os.path.exists("trained_emotion_model.pkl"):
    print("🚀 Training model on RAVDESS dataset...")
    train_model()
else:
    print("✅ Model already trained. Skipping training.")

# 📌 Load Trained Model
def load_trained_model():
    """Loads the trained model and encoder from disk."""
    MODEL_PATH, ENCODER_PATH = "trained_emotion_model.pkl", "label_encoder.pkl"

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("❌ Error: Model or encoder file not found!")
        exit()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    print("✅ Model and encoder loaded successfully!")
    return model, encoder

model, encoder = load_trained_model()

# 📌 Extract Features for Prediction
def extract_feature(audio_data, sample_rate):
    """Extracts audio features for emotion recognition."""
    result = np.array([])

    # Ensure correct sample rate
    TARGET_SAMPLE_RATE = 44100
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)

    # Combine and normalize features
    result = np.hstack([mfccs, chroma, mel])
    result = StandardScaler().fit_transform(result.reshape(1, -1)).flatten()

    return result

# 📌 Record Audio for Prediction
def record_audio(duration=5, sample_rate=44100):
    """Records real-time audio."""
    print(f"🎤 Recording for {duration} seconds... Please speak.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten(), sample_rate

# 📌 Predict Emotion
def analyze_emotion(audio_path):
    """Predicts emotion from recorded audio and returns probability distribution."""
    
    # Extract features from audio
    features = extract_feature(audio_data, sample_rate).reshape(1, -1)

    # Get probabilities for each emotion
    predicted_proba = model.predict_proba(features)[0]  # Returns an array of probabilities

    # Map probabilities to emotion labels & ensure they are Python floats
    emotion_labels = encoder.classes_  # Get emotion labels from the encoder
    emotion_probs = {label: round(float(prob), 2) for label, prob in zip(emotion_labels, predicted_proba)}

    # Convert probabilities to readable format
    emotion_str = ", ".join([f"{str(emo)}: {float(prob):.2f}" for emo, prob in emotion_probs.items()])

    
    print(f"🔹 Detected Emotions: {emotion_str}")

    return emotion_probs


# def predict_emotion(audio_data, sample_rate):
    
#     """Predicts emotion from recorded audio."""
#     features = extract_feature(audio_data, sample_rate).reshape(1, -1)
#     predicted_proba = model.predict_proba(features)[0]
#     predicted_label = np.argmax(predicted_proba)
#     emotion = encoder.inverse_transform([predicted_label])[0]

#     print(f"🎭 Detected Emotion: {emotion}")
#     return emotion

# 📌 Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📌 Generate ChatGPT Response Based on Emotion
def generate_chatgpt_response(messages):
    """Starts a GPT conversation based on detected emotion."""
    # messages = [
    #     # {"role": "system", "content": "You are a friendly and empathetic assistant. Start a conversation based on user emotion."},
    #     # {"role": "user", "content": f"I'm feeling {emotion}. Let's talk about it."}
    # ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return "Sorry, I encountered an error generating a response."

# 📌 Main Program Loop
if __name__ == "__main__":
    print("🎙️ Real-Time Speech Emotion Detection + ChatGPT")
    print("Press Ctrl+C to exit the program.\n")

    while True:
        # Step 1: Detect Emotion
        audio_data, sample_rate = record_audio(duration=5)

        if audio_data is None or sample_rate is None:
            print("❌ Error: Unable to record audio. Please try again.")
            continue

        emotion = predict_emotion(audio_data, sample_rate)

        if emotion:
            print(f"🧠 Are you feeling {emotion}?")

            # Step 2: Start a New Conversation Based on Emotion
            messages = [
                {"role": "system", "content": "You are trying to help users process their emotions. Guess what emotions the user is feeling based on what the user is saying."},
                {"role": "user", "content": f"Are you feeling {emotion}? I want to understand you better."}
            ]

            # Get GPT's first response
            chat_response = generate_chatgpt_response(messages)
            print("\n--- ChatGPT's Response ---\n")
            print(chat_response)
            print("--------------------------\n")

            # Save GPT's response to conversation history
            messages.append({"role": "assistant", "content": chat_response})

            # Step 3: Let the User Continue the Conversation
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting... Goodbye!")
                    break

                messages.append({"role": "user", "content": user_input})

                chat_response = generate_chatgpt_response(messages)
                print("\n--- ChatGPT's Response ---\n")
                print(chat_response)
                print("--------------------------\n")

                messages.append({"role": "assistant", "content": chat_response})

        else:
            print("❌ Could not detect emotion. Please try again.")


import gradio as gr
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 🔹 Load trained model and encoder
with open("trained_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# 🔹 Define function to predict emotion
def predict_emotion(audio):
    try:
        # Load audio file
        y, sr = librosa.load(audio, sr=44100)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        # Combine and normalize features
        features = np.hstack((mfccs, chroma, mel)).reshape(1, -1)
        features = StandardScaler().fit_transform(features)

        # Predict emotion
        predicted_label = model.predict(features)[0]
        emotion = encoder.inverse_transform([predicted_label])[0]

        return f"Detected Emotion: {emotion} 😊"

    except Exception as e:
        return f"Error: {str(e)}"

# 🔹 Create Gradio UI
iface = gr.Interface(
    fn=predict_emotion,
    inputs="audio",
    outputs="text",
    title="🎤 Emotion Detector",
    description="Record or upload an audio file, and the AI will detect the emotion!",
    live=True,
)

