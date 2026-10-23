from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # copies OPENAI_API_KEY from .env into the environment, where OpenAI() looks for it
client = OpenAI()
response = client.responses.create(model="gpt-4o-mini", input="hello")
print(response.output_text)
