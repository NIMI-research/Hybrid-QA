from Tools.Tool import Squall, SparqlTool, WikiTool
from Tools.utilities_for_tools import load_refined_model, load_chain
import json
import fire
from Lang_file import Lanchain_impl_wikipedia
import os
import re
from langchain import PromptTemplate
from langchain import LLMChain
from utils import extract_values, read_json, write_answers, prepare_question_list
import datetime
import os
import logging
import torch, gc
import time






def main(
    dataset: str = "mintaka",
    model_name: str = "gpt-4-0314",
    output_path: str = "late_fusion",
    few_shot_dataset = "mintaka_wikipedia"
):
    logging.info(
        f"------Dataset: {dataset}, Model: {model_name}"
    )
    wiki_tool = WikiTool(model_name)
    path = os.getcwd()
    print("main---->", path)
    questions = prepare_question_list(dataset)
    print(questions)
    langchain_call = Lanchain_impl_wikipedia(
        dataset, model_name, wiki_tool, few_shot_dataset
    )
    print("Here---Bro")
    final_answer_list = []
    for idx, question in enumerate(questions):
        temp = {}
        try:
            logging.info(
                f"----------Evaluation on Question: {question} Index: {idx}----------"
            )
            temp["question"] = question
            out,template_answer = langchain_call.execute_agent_wikipedia(question.strip("\n"))
            temp["wikipedia_answer"] = out.get("output")
            temp["error"] = None
            temp["intermediate_logs"] = template_answer
            final_answer_list.append(temp)
            logging.info(temp)
            logging.info(
                f"----Evaluation Done Question: {question} Index: {idx}---"
            )
        except Exception as e:
            temp["question"] = question
            temp["final_answer"] = None
            temp["intermediate_logs"] = None
            temp["error"] = str(e)
            final_answer_list.append(temp)
        del temp
        if (idx+1) % 2 == 0:
            write_answers(final_answer_list, output_path, dataset)
        continue

if __name__ == "__main__":
    fire.Fire(main)