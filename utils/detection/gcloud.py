import os
from io import BytesIO

from google.cloud import vision
from google.oauth2 import service_account
from google.api_core.exceptions import (
    GoogleAPICallError,
    InvalidArgument
)
from PIL import Image

client = vision.ImageAnnotatorClient(credentials=service_account.Credentials.from_service_account_file(os.getenv("GCLOUD_CREDENTIALS_PATH")))

def payload_process(payload) -> dict:
    data = {
        "data": []
    }
    for index in range(len(payload.text_annotations)):
        data["data"].append({
            "text": payload.text_annotations[index].description,
            "vertex": {
                "tl": [payload.text_annotations[index].bounding_poly.vertices[0].x, payload.text_annotations[index].bounding_poly.vertices[0].y],
                "tr": [payload.text_annotations[index].bounding_poly.vertices[1].x, payload.text_annotations[index].bounding_poly.vertices[1].y],
                "bl": [payload.text_annotations[index].bounding_poly.vertices[3].x, payload.text_annotations[index].bounding_poly.vertices[3].y],
                "br": [payload.text_annotations[index].bounding_poly.vertices[2].x, payload.text_annotations[index].bounding_poly.vertices[2].y]
            }
        })
    return data

def run(img:bytes|str|Image.Image) -> dict:
    if isinstance(img, str):
        content = open(img, "rb").read()
    elif isinstance(img, bytes):
        content = img
    elif isinstance(img, Image.Image):
        with BytesIO() as byte_buffer:
            img.save(byte_buffer, format="JPEG")
            content = byte_buffer.getvalue()
    else:
        raise ValueError("Unsupported file type, only bytes, str, or Image.Image types are allowed")
    
    image = vision.Image(content=content)
    
    try:
        response = self.client.text_detection(image=image)

    except GoogleAPICallError as e:
        raise GoogleAPICallError(
            f"API call error: {e.message}\n"
            "For more info on error messages, check: \n"
            "https://cloud.google.com/apis/design/errors"
        )

    except InvalidArgument as e:
        raise InvalidArgument(
            f"Invalid argument: {e.message}\n"
            "For more info on error messages, check: \n"
            "https://cloud.google.com/apis/design/errors"
        )
    
    except Exception as e:
        raise Exception(
            f"Unknown error: {e.message}\n"
            "For more info on error messages, check: \n"
            "https://cloud.google.com/apis/design/errors"
        )

    if response.error.message:
        raise GoogleAPICallError(
            f"{response.error.message}\n"
            "For more info on error messages, check: \n"
            "https://cloud.google.com/apis/design/errors"
        )

    return payload_process(response)
