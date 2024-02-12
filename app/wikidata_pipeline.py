from Tools.Tool import Squall, SparqlTool, WikiTool
from Tools.utilities_for_tools import load_refined_model, load_chain
import json
import fire
from Lang_file import Lanchain_impl_wikidata
import os
import re
from langchain import PromptTemplate
from langchain import LLMChain
from utils import extract_values, read_json, write_answers, prepare_question_list
import datetime
import os
import torch, gc
import time
import logging


def main(
    dataset: str = "compmix",
    model_name: str = "gpt-4-0314",
    output_path: str = "late_fusion",
    dynamic=False,
    few_shot_data: str = "compmix_wikidata",
):
    logging.info(f"------FUSION Dataset: {dataset}, Model: {model_name}")
    refined = load_refined_model()
    path = os.getcwd()
    wiki_tool = WikiTool(model_name)
    path = os.getcwd()
    print("main---->", path)
    squall = Squall(
        f"{path}/Tools/Tools_Data/squall_fixed_few_shot.json",
        refined,
        model_name,
        wikidata=True,
    )
    sparql_tool = SparqlTool(f"{path}/Tools/Tools_Data/squall2sparql_revised.sh")
    questions = prepare_question_list(dataset)
    langchain_call = Lanchain_impl_wikidata(
        few_shot_data, model_name, wiki_tool, squall, sparql_tool, dynamic
    )
    final_answer_list = []
    for idx, question in enumerate(questions):
        temp = {}
        try:
            time.sleep(30)
            logging.info(
                f"----------Evaluation on Question: {question} Index: {idx}----------"
            )
            temp["question"] = question
            out, template_answer = langchain_call.execute_agent(question.strip("\n"))
            temp["wikipedia_answer"] = out.get("output")
            temp["error"] = None
            temp["intermediate_logs"] = template_answer
            final_answer_list.append(temp)
            logging.info(f"----Evaluation Done Question: {question} Index: {idx}---")
        except Exception as e:
            temp["question"] = question
            temp["final_answer"] = None
            temp["intermediate_logs"] = None
            temp["error"] = str(e)
            final_answer_list.append(temp)
        del temp
        if (idx + 1) % 10 == 0:
            write_answers(final_answer_list, output_path, dataset)
        continue


if __name__ == "__main__":
    fire.Fire(main)
