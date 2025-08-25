# Performance Configuration
import os

# OpenAI Configuration
OPENAI_MAX_TOKENS_JSON = 2000
OPENAI_MAX_TOKENS_HTML = 3000
OPENAI_MODEL = "gpt-4o"

# Cache Configuration
ENABLE_CACHING = True
CACHE_TTL = 3600  # 1 hour in seconds

# Async Configuration
MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMEOUT = 120  # 2 minutes

# Image Processing
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
MAX_IMAGE_SIZE_MB = 20

# API Configuration
API_RATE_LIMIT = "10/minute"