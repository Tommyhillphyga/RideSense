
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from speechlistener.audiolistener import AudioListener


def main():
    """
    Main function to run the Phrase listener
    """

    listener = AudioListener(
        similarity_threshold=0.5,
        min_word_length= 3
    )

    logger.info(f"Real-time listening (requires microphone) \n")
    listener.listen()


if __name__ == "__main__":
    main()