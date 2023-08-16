from langchain.chat_models import ChatOpenAI
import configparser
import os
from sentence_transformers import SentenceTransformer, util

def load_chain():
    llm = ChatOpenAI(model_name='gpt-4-0314', temperature=0)
    return llm

def load_openai_api():
    config = configparser.ConfigParser()
    config.read('/home/dhananjay/HybridQA/Tools/Tools_Data/secrets.ini')
    openai_api_key = config['OPENAI']['OPENAI_API_KEY']
    os.environ.update({'OPENAI_API_KEY': openai_api_key})
    print(openai_api_key)
    return config

def load_sentence_transformer():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model
