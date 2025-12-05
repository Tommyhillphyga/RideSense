
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from speechlistener.audiolistener import AudioListener


def main(model_path: str = "models/vosk-model-small-en-us-0.15"):
    """
    Main function to run the Phrase listener
    """

    listener = AudioListener(
        model_path= model_path,
        similarity_threshold=0.5,
        min_word_length= 3
    )

    logger.info(f"Real-time listening (requires microphone) \n")
    listener.listen()


if __name__ == "__main__":
    models_path = "models/vosk-model-small-en-us-0.15"
    main(model_path=models_path)