#******************************************#
# Application:
# Created:
#******************************************#
import os
import openai
from API_KEYS_ML import openAI_API_key

print("CONSOLE CHAT-GPT")

openai.api_key = openAI_API_key
response = openai.Completion.create(
  model="text-davinci-003",
  prompt= input("What is your question?\n"),
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\"\"\""]
)

answer: str = response["choices"][-1]["text"]

print("ANSWER\n" + answer)
