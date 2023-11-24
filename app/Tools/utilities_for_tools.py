from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import configparser
import os
from sentence_transformers import SentenceTransformer, util
from refined.inference.processor import Refined
import os


def load_chain(model_name):
    if model_name in ['gpt-4-0314','gpt-3.5-turbo']:
        llm = ChatOpenAI(model_name = model_name, temperature=0,request_timeout=300)
    else:
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 100},
            model_kwargs=dict({"trust_remote_code":True})
        )
    return llm

def load_openai_api():
    config = configparser.ConfigParser()
    path = os.getcwd()
    config.read(f'{path}/Tools/Tools_Data/secrets.ini')
    openai_api_key = config['OPENAI']['OPENAI_API_KEY']
    os.environ.update({'OPENAI_API_KEY': openai_api_key})
    print(openai_api_key)
    return config

def load_sentence_transformer():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model


def load_refined_model(refined_cache_dir='~/.cache/refined/'):
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia",
                                      data_dir=refined_cache_dir)
    return refined
