import os
import json
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Debug output to check if API key is loaded
print(f"API Key status: {'Loaded' if openai.api_key else 'Missing'}")
if not openai.api_key:
    print("WARNING: No OpenAI API key found. Emotion detection will use inferential classifier.")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# In-memory conversation store
conversations = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text to classify"}), 400

    print(f"Processing text: '{text}'")

    # If no API key is available, use an inferential fallback
    if not openai.api_key:
        return inferential_classify(text)

    # Improved system prompt for inferring emotions without explicit statements
    system_prompt = """
You are an expert emotion classifier that can detect subtle emotional cues in speech.
Analyze the text and infer the speaker's emotional state, even when emotions aren't
explicitly stated. Look for:

1. Content and context clues (what they're describing)
2. Word choice and intensity markers
3. Sentence structure and phrasing patterns
4. Implied emotional undercurrents

Classify into one of these categories: Happy, Sad, Angry, Fearful, Anxious, Excited, 
Neutral, Surprised, Disgusted, Confused, Tired, or Hungry.

Also assign an intensity value from -100 (extremely negative) to +100 (extremely positive).

Respond ONLY with a JSON object in this format:
{"emotion":"Category","intensity":value}
"""

    try:
        print("Making OpenAI API request...")
        # Try to use the new OpenAI client API format
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            resp = client.chat.completions.create(
                model="gpt-4" if openai.api_key else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text: \"{text}\""}
                ],
                temperature=0.0,
                max_tokens=50
            )
            raw = resp.choices[0].message.content.strip()
        except AttributeError:
            # Fall back to the old API format if necessary
            resp = openai.ChatCompletion.create(
                model="gpt-4" if openai.api_key else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text: \"{text}\""}
                ],
                temperature=0.0,
                max_tokens=50
            )
            raw = resp.choices[0].message.content.strip()
            
        print("OpenAI response received.")
        print(f"Raw GPT response: {raw}")
        
        try:
            result = json.loads(raw)
            emotion = result.get("emotion", "Neutral")
            intensity = int(result.get("intensity", 0))
            print(f"Successfully parsed emotion: {emotion}, intensity: {intensity}")
            
            # Map emotion to confidences for the p5.js visualization
            result = map_emotion_to_confidences(emotion, intensity)
            print(f"Returning: {result}")
            return jsonify(result)
            
        except Exception as e:
            print(f"Failed to parse JSON from GPT: {raw}, Error: {e}")
            # If parsing fails, use the inferential classifier as fallback
            return inferential_classify(text)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        # If API call fails, use the inferential classifier
        return inferential_classify(text)

def map_emotion_to_confidences(emotion, intensity):
    """Map the detected emotion and intensity to confidence values for visualization"""
    # Expanded emotion mapping including Tired and Hungry categories
    emotion_map = {
        "Happy": {"sad": 0, "fear": 0, "anger": 0, "anxiety": 0, "neutral": 0.2, "excitement": 0.3, "joy": 0.5},
        "Sad": {"sad": 0.7, "fear": 0.1, "anger": 0, "anxiety": 0.2, "neutral": 0, "excitement": 0, "joy": 0},
        "Angry": {"sad": 0, "fear": 0, "anger": 0.8, "anxiety": 0.2, "neutral": 0, "excitement": 0, "joy": 0},
        "Fearful": {"sad": 0.1, "fear": 0.7, "anger": 0, "anxiety": 0.2, "neutral": 0, "excitement": 0, "joy": 0},
        "Anxious": {"sad": 0.1, "fear": 0.3, "anger": 0, "anxiety": 0.6, "neutral": 0, "excitement": 0, "joy": 0},
        "Excited": {"sad": 0, "fear": 0, "anger": 0, "anxiety": 0, "neutral": 0.1, "excitement": 0.8, "joy": 0.1},
        "Neutral": {"sad": 0, "fear": 0, "anger": 0, "anxiety": 0, "neutral": 1, "excitement": 0, "joy": 0},
        "Surprised": {"sad": 0, "fear": 0.2, "anger": 0, "anxiety": 0.1, "neutral": 0.1, "excitement": 0.6, "joy": 0},
        "Disgusted": {"sad": 0.1, "fear": 0.1, "anger": 0.6, "anxiety": 0.2, "neutral": 0, "excitement": 0, "joy": 0},
        "Confused": {"sad": 0.1, "fear": 0.2, "anger": 0.1, "anxiety": 0.4, "neutral": 0.2, "excitement": 0, "joy": 0},
        "Tired": {"sad": 0.3, "fear": 0, "anger": 0.1, "anxiety": 0.2, "neutral": 0.4, "excitement": 0, "joy": 0},
        "Hungry": {"sad": 0.1, "fear": 0, "anger": 0.2, "anxiety": 0.3, "neutral": 0.4, "excitement": 0, "joy": 0}
    }
    
    # Get confidences for the detected emotion or default to neutral
    confidences = emotion_map.get(emotion, emotion_map["Neutral"])
    
    # Adjust confidences based on intensity
    intensity_factor = abs(intensity) / 100
    if intensity > 0:  # Positive emotions
        confidences["joy"] = confidences.get("joy", 0) * intensity_factor
        confidences["excitement"] = confidences.get("excitement", 0) * intensity_factor
    else:  # Negative emotions
        confidences["sad"] = confidences.get("sad", 0) * intensity_factor
        confidences["fear"] = confidences.get("fear", 0) * intensity_factor
        confidences["anger"] = confidences.get("anger", 0) * intensity_factor
        confidences["anxiety"] = confidences.get("anxiety", 0) * intensity_factor
    
    return {
        "emotion": emotion, 
        "intensity": intensity,
        "confidences": confidences
    }

def inferential_classify(text):
    """More sophisticated context-based emotion classifier for when OpenAI API is unavailable"""
    text_lower = text.lower()
    
    print(f"Using inferential classifier for: '{text}'")
    
    # Initialize score counters for each emotion
    scores = {
        "Happy": 0,
        "Sad": 0,
        "Angry": 0,
        "Anxious": 0,
        "Fearful": 0,
        "Excited": 0,
        "Neutral": 0.5,  # Slight bias for neutral as default
        "Surprised": 0,
        "Disgusted": 0,
        "Confused": 0,
        "Tired": 0,     # New category
        "Hungry": 0     # New category
    }
    
    # Enhanced contextual detection patterns with additional categories
    patterns = {
        "Happy": [
            "great", "wonderful", "fantastic", "amazing", "good", "love", "awesome", 
            "enjoy", "pleased", "delighted", "win", "success", "accomplished", 
            "birthday", "celebrate", "proud", "perfect", "beautiful", "sunshine",
            "excited about", "looking forward", "can't wait", "fun"
        ],
        "Sad": [
            "sad", "down", "unhappy", "depressed", "miserable", "hurt", "pain", 
            "lonely", "alone", "miss", "lost", "sorry", "regret", "cry", "tear",
            "heartbroken", "disappointed", "grief", "upset", "funeral", "died",
            "miss home", "homesick", "exhausted", "worn out"
        ],
        "Angry": [
            "angry", "mad", "furious", "upset", "irritated", "annoyed", "frustrated",
            "hate", "unfair", "ridiculous", "blame", "fault", "stupid", "idiot", 
            "terrible", "worst", "ruined", "horrible", "hell", "damn", "fed up",
            "sick of", "tired of", "had enough"
        ],
        "Anxious": [
            "anxious", "nervous", "worried", "stress", "pressure", "overwhelmed",
            "afraid", "fear", "panic", "uncertain", "doubt", "risk", "concern",
            "interview", "test", "exam", "deadline", "meeting", "presentation",
            "want to go home", "need to leave", "can't stay", "have to go"
        ],
        "Tired": [
            "tired", "exhausted", "sleepy", "fatigue", "drained", "no energy",
            "worn out", "need sleep", "need rest", "need a break", "can't keep going",
            "so tired", "want to sleep", "want to rest", "want to go home", "need to lie down",
            "long day", "hard day", "ready for bed", "eyes heavy"
        ],
        "Hungry": [
            "hungry", "starving", "need food", "want to eat", "need to eat", "food",
            "haven't eaten", "stomach growling", "stomach rumbling", "need a meal", 
            "want a snack", "dinner", "lunch", "breakfast", "craving", "appetite"
        ],
        "Fearful": [
            "scared", "terrified", "horrified", "danger", "threat", "attack",
            "nightmare", "monster", "dark", "alone", "unknown", "help", "run", "hide",
            "scream", "horror", "killer", "death", "dying", "terror", "emergency"
        ],
        "Excited": [
            "excited", "thrilled", "eager", "looking forward", "cant wait", "anticipate",
            "adventure", "fun", "party", "vacation", "holiday", "weekend", "opportunity",
            "chance", "new", "start", "beginning", "future", "potential", "possibility"
        ],
        "Surprised": [
            "surprised", "shocked", "unexpected", "wow", "whoa", "amazing", "unbelievable",
            "incredible", "what", "how", "suddenly", "no way", "impossible", "cant believe",
            "believe it", "really", "serious", "never thought", "never expected"
        ],
        "Disgusted": [
            "disgusting", "gross", "sick", "nasty", "eww", "vomit", "rotten", "filthy",
            "dirty", "ugly", "horrible", "worst", "unacceptable", "terrible", "creepy"
        ],
        "Confused": [
            "confused", "unsure", "dont understand", "lost", "complicated", "complex",
            "what do you mean", "unclear", "not sure", "dont get it", "strange",
            "weird", "bizarre", "odd", "wonder", "question", "how", "why", "when",
            "don't know why", "don't even know"
        ],
        "Neutral": [
            "okay", "fine", "alright", "normal", "regular", "usual", "so-so", "meh"
        ]
    }
    
    # Check for emotion keywords in the text
    for emotion, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[emotion] += 1
    
    # Handle negations
    negation_words = ["not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"]
    for negation in negation_words:
        if negation in text_lower:
            # Find words after negation
            negation_idx = text_lower.find(negation)
            negated_part = text_lower[negation_idx:]
            
            # Check if emotions are being negated
            for emotion, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in negated_part:
                        # Reduce that emotion's score
                        scores[emotion] -= 1
                        # Potentially increase opposite emotions
                        if emotion == "Happy":
                            scores["Sad"] += 0.5
                        elif emotion == "Sad":
                            scores["Happy"] += 0.5
                        elif emotion == "Tired":
                            scores["Excited"] += 0.5
                        elif emotion == "Excited":
                            scores["Tired"] += 0.5

    # Add special phrase handling
    phrases = {
        "want to go home": {"Tired": 2, "Anxious": 1},
        "need to go home": {"Tired": 2, "Anxious": 1},
        "long day": {"Tired": 2},
        "so hungry": {"Hungry": 3},
        "so tired": {"Tired": 3},
        "don't know why": {"Confused": 2},
        "don't even know": {"Confused": 2},
        "just like really want": {"Anxious": 1.5, "Tired": 1},
        "been a long day": {"Tired": 2.5},
        "had a really long day": {"Tired": 3},
        "feeling happy": {"Happy": 3},
        "feeling sad": {"Sad": 3},
        "feeling angry": {"Angry": 3},
        "feeling tired": {"Tired": 3},
        "feeling hungry": {"Hungry": 3},
        "feeling anxious": {"Anxious": 3},
        "feeling confused": {"Confused": 3},
        "don't even know why": {"Confused": 3}
    }

    for phrase, emotion_scores in phrases.items():
        if phrase in text_lower:
            for emotion, score in emotion_scores.items():
                scores[emotion] += score
    
    # Context-based inferences (beyond simple keyword matching)
    # Life events
    if any(phrase in text_lower for phrase in ["got a promotion", "graduated", "passed my test", "got the job", "won", "won the"]):
        scores["Happy"] += 2
        scores["Excited"] += 1
        
    if any(phrase in text_lower for phrase in ["lost my", "broke up", "failed", "missed", "too late", "never get to"]):
        scores["Sad"] += 2
        
    if any(phrase in text_lower for phrase in ["deadline", "running late", "not enough time", "have to finish", "due tomorrow"]):
        scores["Anxious"] += 2
        
    # Physical symptoms
    if any(phrase in text_lower for phrase in ["cant sleep", "heart racing", "shaking", "trembling", "sweat", "sweating"]):
        scores["Anxious"] += 2
        scores["Fearful"] += 1
        
    if any(phrase in text_lower for phrase in ["yelled", "screamed", "threw", "broke", "hit", "slammed", "cursed"]):
        scores["Angry"] += 2
    
    # Specific for tiredness and hunger
    if any(phrase in text_lower for phrase in ["need a nap", "could sleep for days", "barely keeping eyes open", "so sleepy"]):
        scores["Tired"] += 3
        
    if any(phrase in text_lower for phrase in ["stomach growling", "haven't eaten all day", "need to eat soon", "starving"]):
        scores["Hungry"] += 3
    
    # Weather and situational context
    if any(phrase in text_lower for phrase in ["beautiful day", "sunny", "perfect weather", "lovely outside"]):
        scores["Happy"] += 1
        
    if any(phrase in text_lower for phrase in ["rainy", "dark", "gloomy", "alone in"]):
        scores["Sad"] += 1
    
    # Tone indicators
    if "!" in text:
        exclamation_count = text.count("!")
        if exclamation_count >= 3:
            scores["Excited"] += 2
            scores["Happy"] += 1
            scores["Surprised"] += 1
        elif exclamation_count >= 1:
            scores["Excited"] += 1
            scores["Happy"] += 0.5
    
    if "?" in text:
        question_count = text.count("?")
        if question_count >= 3:
            scores["Confused"] += 2
            scores["Anxious"] += 1
        elif question_count >= 1:
            scores["Confused"] += 0.5
    
    # Intensity markers
    intensifiers = ["very", "really", "extremely", "so", "totally", "absolutely", "completely", "utterly", "super"]
    intensity_boost = 0
    for intensifier in intensifiers:
        if intensifier in text_lower:
            intensity_boost += 0.2
    
    # Apply intensity boost to the highest emotions
    if intensity_boost > 0:
        top_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        for emotion, _ in top_emotions:
            scores[emotion] += intensity_boost
    
    # If no strong emotion detected, strengthen neutral
    if max(scores.values()) < 1:
        scores["Neutral"] = 2
    
    # Find the predominant emotion
    emotion = max(scores, key=scores.get)
    
    # Set intensity based on the emotion and score
    base_intensities = {
        "Happy": 60, "Excited": 70, "Surprised": 40,  # Positive emotions
        "Neutral": 0,  # Neutral
        "Sad": -60, "Angry": -70, "Anxious": -40, "Tired": -30, "Hungry": -20,
        "Fearful": -50, "Disgusted": -60, "Confused": -20  # Negative emotions
    }
    
    # Adjust intensity based on score strength
    score_factor = min(2.0, scores[emotion]) / 2.0  # Cap at 2.0 to avoid extremes
    intensity = int(base_intensities[emotion] * score_factor * 1.2)  # Scale up slightly
    
    # Ensure intensity is within bounds
    intensity = max(-100, min(100, intensity))
    
    print(f"Inferential detection: {emotion}, intensity: {intensity}")
    print(f"Emotion scores: {scores}")
    
    # Map emotion to confidences and return
    return jsonify(map_emotion_to_confidences(emotion, intensity))

@app.route("/respond", methods=["POST"])
def respond():
    data = request.get_json()
    emotion = data.get("emotion", "Neutral")
    text    = data.get("text", "")
    prompt = (
      f"You are a compassionate assistant. The user is feeling {emotion}. "
      f"They said: \"{text}\". Reply in one or two sentences showing empathy."
    )
    try:
        # Try to use the new OpenAI client API format first
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=60
            )
            reply = resp.choices[0].message.content.strip()
        except AttributeError:
            # Fall back to the old API format if necessary
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=60
            )
            reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in respond endpoint: {e}")
        reply = f"I understand you're feeling {emotion.lower()}. How can I help you today?"

    # Start a new chat session
    chat_id = uuid.uuid4().hex
    conversations[chat_id] = [
      {"role": "system", "content": "You are a compassionate assistant."},
      {"role": "assistant", "content": reply}
    ]
    return jsonify({"message": reply, "chat_id": chat_id})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    chat_id = data.get("chat_id")
    user_msg = data.get("message", "").strip()
    if not chat_id or chat_id not in conversations:
        return jsonify({"error": "Invalid chat_id"}), 400

    conversations[chat_id].append({"role": "user", "content": user_msg})
    try:
        # Try to use the new OpenAI client API format first
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversations[chat_id] + [{"role": "user", "content": user_msg}],
                temperature=0.7,
                max_tokens=60
            )
            assistant_msg = resp.choices[0].message.content.strip()
        except AttributeError:
            # Fall back to the old API format if necessary
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversations[chat_id] + [{"role": "user", "content": user_msg}],
                temperature=0.7,
                max_tokens=60
            )
            assistant_msg = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        assistant_msg = "I'm having trouble connecting right now. Can we try again in a moment?"
    
    conversations[chat_id].append({"role": "assistant", "content": assistant_msg})
    return jsonify({"reply": assistant_msg})

@app.route('/ping')
def ping():
    return 'pong'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)  # Changed port to 5000 to match your browser URL