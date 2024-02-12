import re
import os 
import json
import datetime

def extract_values(data_str):
    wikipedia_match = re.search(r'wikipedia_answer\s*:\s*(.*?)(?=\s*,\s*wikidata_answer|$)', data_str, re.I)
    wikipedia_value = wikipedia_match.group(1).strip() if wikipedia_match else None
    if wikipedia_value and wikipedia_value.endswith(','):
        wikipedia_value = wikipedia_value[:-1].strip()

    wikidata_match = re.search(r'wikidata_answer\s*:\s*(?:\[(.*?)\]|([^[]+))', data_str, re.I)
    if wikidata_match:
        if wikidata_match.group(1): 
            wikidata_value = '[' + wikidata_match.group(1).strip() + ']'
        else: 
            wikidata_value = wikidata_match.group(2).strip()
    else:
        wikidata_value = None

    return wikipedia_value, wikidata_value


def read_json(dataset):
    path = os.getcwd()
    with open(f"{path}/data/merge_few_shot_examples.json","r") as file:
        data = json.load(file)
        return data[dataset]
    
def write_answers(answer_list, output_path, dataset, answer=True):
    json_write = json.dumps(answer_list, indent=4)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
    path = ""
    if answer:
        path = f"{output_path}/{timestamp}_{dataset}_answer.json" 
    else:
        path = f"{output_path}/{dataset}_templates.json"
    with open(path, "w") as outfile:
        outfile.write(json_write)


def prepare_question_list(data_file):
    with open(f"data/{data_file}.txt", "r") as file:
        lines = file.readlines()
        return lines