import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def generate_request_id():
    return str(uuid.uuid4())
