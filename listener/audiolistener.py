import json
import queue
import threading
import time
import sys
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
# configure logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pyaudio
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
except ImportError as e:
    logger.info(f"Required packages not installed. Please install with: \n")
    logger.info(f"pip install vosk pyaudio sounddevice")
    logger.info(f"Error: {e}")
    sys.exit(1)

@dataclass
class TripEvent:
    "Data class for trip events"
    keyword: str
    timestamp: float
    confidence: float
    matched_phrase: str
    original_text: str


class AudioListener:
    "A lightweight speech-to-text system for detecting trip keywords."
    def __init__(
            self, 
            model_path: Optional[str] = None, 
            sample_rate: int = 16000,
            similarity_threshold: float = 0.6, 
            min_word_length: int = 3
            ):
        """
        Initialize the AudioListener 

        Args:
            model_path (str): Path to the Vosk model directory.
            sample_rate (int): Audio sample rate in HZ.
        """ 

        self.sampe_rate = sample_rate
        self.keyword_queue = queue.Queue()
        self.is_listening = False
        self.similarity_threshold = similarity_threshold
        self.min_word_length = min_word_length

        # Define keyworks to listen for 
        with open("keyword_phrases.json", "r") as f:
            self.keywords = json.load(f)

        #Flatten all phrases into a single list
        self.all_phrases = []
        self.phrases_to_category = {}

        for category, phrases in self.keywords.items():
            self.all_phrases.extend(phrases)
            self.phrases_to_category.update({phrase: category for phrase in phrases})

        
        #Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )

        logger.info(f"Fitting TF-IDF vectorizer on {len(self.all_phrases)} phrases.")
        self.phrase_vectors = self.vectorizer.fit_transform(self.all_phrases)

        # Load Vosk model
        if model_path is None:
            model_path = "vosk-model-small-en-us-0.15"
            logger.info(f"No model path provided. Using default: {model_path}")
            logger.info(f"Loading Vosk model from: {model_path}")

        try:
            self.model = Model(model_path)
        except Exception as e:
            logger.error(f"Failed to load Vosk model from {model_path}. Error: {e}")
            sys.exit(1)

        self.recognizer = KaldiRecognizer(self.model, self.sampe_rate)

        #statistic tracking
        self.detection_stat = {
            'total_utterances': 0,
            'keywords_detected': 0,
            'false_positives': 0,
            'similarity_scores': []
        }


    def _preprocess_text(
            self,
            text: str
    ) -> str:
        """
            Normalise text for keyword matching
        
        """
        text = text.lower().strip()
        
        filler_words = ['um', 'uh', 'er', 'ah' 'like', 'you know', 'so']

        #Remove filler words
        for filler in filler_words:
            text = text.replace(f"{filler} "," ")

        #Remove extra whitespaces
        words = [word for word in text.split() if len(text) >= self.min_word_lengthmin_]
        return ' '.join(words)

    def _find_best_match(self, query_text: str) -> Optional[Tuple[str, str, float]]:
        pass
         "added ds"
    
        

        
        
        
    
    
    
        



