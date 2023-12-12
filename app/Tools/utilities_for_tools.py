from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms import VLLM
import configparser
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from refined.inference.processor import Refined
import os
import torch

def load_chain(model_name,use_vllm):
    if model_name in ['gpt-4-0314','gpt-3.5-turbo']:
        llm = ChatOpenAI(model_name = model_name, temperature=0,request_timeout=300)
    
    elif use_vllm:
        print("Loading Model using VLLM")

        if not torch.cuda.is_available():
            raise Exception("vLLM currently only supports GPU Inference but cuda is not available.")
        
        devices = torch.cuda.device_count()

        llm = VLLM(model=model_name,
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=200,
           top_k=10,
           top_p=0.95,
           temperature=0.8,
           tensor_parallel_size=devices # for distributed inference
        )       
    else:
        #loading model via standard HF Pipeline
        print('Loading Model using Huggingface Pipeline')
        model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = 0 if torch.cuda.is_available() else -1
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=100,device=device)
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

