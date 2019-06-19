import os

# ENV = {
#     "TFHUB_WORD_MODEL_DIR": os.environ.get("TFHUB_WORD_MODEL_DIR"),
#     "DB_URL_BOT": os.environ.get("DB_URL_BOT"),
#     "TFHUB_SENTENCE_MODEL_DIR": os.environ.get("TFHUB_SENTENCE_MODEL_DIR")
# }

ENV = {
    "DB_URL_BOT": "mongodb://172.30.24.47:27017",
    "TFHUB_SENTENCE_MODEL_DIR": "/home/atom/Git/embeddings/sentence_model/",
    "TRAINING_DATA":'/home/atom/UST/git/SentenceSimilarity/data/training_data.txt',
    "MONGO_HOST":"mongodb://172.30.24.47:27017",
    "MONGO_DB":'ust_db'
}
