import os

ENV = {
    "TFHUB_WORD_MODEL_DIR": os.environ.get("TFHUB_WORD_MODEL_DIR"),
    "DB_URL_BOT": os.environ.get("DB_URL_BOT"),
    "TFHUB_SENTENCE_MODEL_DIR": os.environ.get("TFHUB_SENTENCE_MODEL_DIR")
}

# ENV = {
#     "DB_URL_BOT": "mongodb://192.168.2.67:27017",
#     "TFHUB_WORD_MODEL_DIR": "D:/Kanverse Project/word_model",
#     "TFHUB_SENTENCE_MODEL_DIR": "D:/Kanverse Project/sentence_model"
# }
