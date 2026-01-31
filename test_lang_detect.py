
from src.nlp_engine import IntentClassifier
import logging

# Mute logging for the test
logging.basicConfig(level=logging.ERROR)

def test_language_detection():
    print("Initializing Classifier...")
    classifier = IntentClassifier()
    
    test_cases = [
        ("Hello, how are you?", "en"),
        ("Hola, como estas?", "es"),
        ("Bonjour, comment ca va?", "fr"),
        ("Hallo, wie geht es dir?", "de")
    ]
    
    print("\nRunning Tests:")
    for text, expected in test_cases:
        lang = classifier.detect_language(text)
        print(f"Text: '{text}' -> Detected: {lang} (Expected: {expected})")
        # Note: The model might return 'en' for English, 'es' for Spanish etc. 
        # We'll just print it to verify.

if __name__ == "__main__":
    test_language_detection()
