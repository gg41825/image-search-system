MONGO_URI = "mongodb://root:example@localhost:27017"
MONGO_DB_NAME = "product_db"

MODEL_BERT = "bert-base-uncased"
MODEL_DINO = "dinov2_vitb14"
# SAMPLE_SIZE must be greater than 0
SAMPLE_SIZE = 10

# (Optional) Triton Inference Server REST endpoint
TRITON_URL = "http://localhost:8000"
TRITON_MODEL_NAME = "aligned"

# (Optional) Vector dimensions (adjusted based on your exported CLIP encoder)
EMBEDDING_DIM = 512