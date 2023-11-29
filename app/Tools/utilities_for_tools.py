from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import configparser
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from refined.inference.processor import Refined
import os
import torch


def load_chain(model_name):
    if model_name in ['gpt-4-0314','gpt-3.5-turbo']:
        llm = ChatOpenAI(model_name = model_name, temperature=0,request_timeout=300)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = get_device()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=30,device=device)
        llm = HuggingFacePipeline(pipeline=pipe)    
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


def get_device():
    return torch.cuda.device_count()-1
