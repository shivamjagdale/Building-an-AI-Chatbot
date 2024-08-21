import os
import json

import google.generativeai as genai
from google.generativeai.embedding import embed_content

working_directory = os.path.dirname(os.path.abspath(__file__))

config_path = f"{working_directory}/config.json"
with open(config_path, "r") as f:
  config_data = json.load(f)

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)


# For Chat Bot
def load_gemini_pro_model():
  gemini_pro_model = genai.GenerativeModel("gemini-pro")
  return gemini_pro_model


# For image recognization
# Loading gemini pro vision model


def load_gemini_pro_vision_model(prompt, image):
  gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash-001")
  response = gemini_pro_vision_model.generate_content([prompt, image])
  result = response.text
  return result


# Now model for embeddings


def embedding_model_gemini(input_text):
  embedding_model = "models/embedding-001"
  embedding = genai.embed_content(model=embedding_model,
                                  content=input_text,
                                  task_type="retrieval_document")
  embedding_list = embedding["embedding"]
  return embedding_list


# Now model for ask me anything :-
def gemini_ask_me_anything(user_prompt):
  gemini_ask_me = genai.GenerativeModel("gemini-pro")
  response = gemini_ask_me.generate_content(user_prompt)
  result = response.text
  return result
