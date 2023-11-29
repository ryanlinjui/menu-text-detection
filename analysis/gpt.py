# -*- coding: utf-8 -*-

import json
from openai import OpenAI
from os import getenv
from dotenv import load_dotenv

from .tools import json_format

class TextAnalysis:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=getenv("OPENAI_API_KEY")
        )

    def run(self, text:str):
        print(text)
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            messages=[{"role": "user", "content": text}],
            tools=[json_format]
        )
        print(json_format)
        return json.loads(completion.choices[0].message.model_dump()["tool_calls"][0]["function"]["arguments"])


text_analysis_api = TextAnalysis()

if __name__ == "__main__":
    menu_text = open("./dataset/data/4.in", "r").read()
    print(menu_text)
    print()
    text_analysis_api.run(menu_text)