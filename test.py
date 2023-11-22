import os

from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI

if __name__ == '__main__':
    load_dotenv()

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY_1"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    history = [
        {
            "role": "user",
            "content": """
Expand[Query]: ['online breast cancer community', 'longitudinal analysis', 'discussion topics', 'convolutional neural networks']
"""
        }
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        messages=history,
        max_tokens=300,
        temperature=1.0,
        top_p=1.0,
    )
    print(response)

