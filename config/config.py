import os

# Maximum query and document lengths
MAX_QUERY_LEN = 96
MAX_DOC_LEN = 512

STORAGE_DIR = os.getenv("STORAGE_DIR", "/media/discoexterno/leon")
# STORAGE_DIR = os.getenv("STORAGE_DIR", "/tmpu/helga_g/leonh_a/qwen-2-vec")
COHERE_API_KEY = "IjmSKOGHCV3LjXLrTWGTKmKzw1N4GfJJ4gyrxJpH"