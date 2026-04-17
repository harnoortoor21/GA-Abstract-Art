from src.midi_processor import extract_emotion_features


midi_file = "src/MIDIs/A-Whole-New-World-(Theme-From-'Aladdin').mid"


features = extract_emotion_features(midi_file)
print(features)