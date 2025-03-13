from dotenv import load_dotenv
import os


def load_secrets(env_path="1.env"):
    # both calls are needed here
    load_dotenv()
    load_dotenv(dotenv_path=env_path)

    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }


load_secrets()


