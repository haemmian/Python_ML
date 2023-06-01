
import os
import openai
from API_KEYS_ML import openAI_API_key
openai.api_key = openAI_API_key


print(openai.Image.create(
  prompt="Space Wallpaper with mars as a main focus",
  n=2,
  size="256x256"
))