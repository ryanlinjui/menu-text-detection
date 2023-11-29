from google.cloud import vision
from google.oauth2 import service_account
from os import getenv
from dotenv import load_dotenv

class ImageDetection:
    def __init__(self):
        load_dotenv()
        credentials = service_account.Credentials.from_service_account_file(
            getenv("GCLOUD_CREDENTIALS_PATH"),
            scopes=['https://www.googleapis.com/auth/cloud-platform'],
        )
        self.client = vision.ImageAnnotatorClient(credentials=credentials)

    def run(self, img_path):
        image_file = open(img_path, "rb").read()
        image = vision.Image(content=image_file)
        response = self.client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        return response.text_annotations[0].description

image_detection_api = ImageDetection()