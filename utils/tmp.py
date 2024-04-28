import uuid
import os
import shutil
from datetime import datetime

def create_tmp_filename(extension:str) -> str:
    return f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_{str(uuid.uuid4())}.{extension}"

def save_tmp_file(file:bytes, extension:str) -> str:
    tmp_filepath = os.path.join("./tmp", create_tmp_filename(extension))
    with open(tmp_filepath, "wb") as f:
        f.write(file)
    return tmp_filepath