from src.nlp_engine import IntentClassifier
import unittest

class TestNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Loading model for testing...")
        cls.nlp = IntentClassifier()

    def test_classification_sales(self):
        text = "I want to buy a subscription"
        labels = ["sales", "support"]
        label, score, lang = self.nlp.classify(text, labels)
        self.assertEqual(label, "sales")
        self.assertEqual(lang, "en")

    def test_classification_support(self):
        text = "My account is locked, please help"
        labels = ["sales", "support"]
        label, score, lang = self.nlp.classify(text, labels)
        self.assertEqual(label, "support")
        self.assertGreater(score, 0.5)

if __name__ == '__main__':
    unittest.main()
