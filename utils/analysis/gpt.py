import json
from os import getenv
from openai import OpenAI

client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

def run(text:str, tools:dict, model:str) -> dict: 
    print(text)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": text}],
        tools=[tools]
    )
    return json.loads(completion.choices[0].message.model_dump()["tool_calls"][0]["function"]["arguments"])