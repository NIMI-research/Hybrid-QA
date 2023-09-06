from Tools.Tool import Squall, SparqlTool, WikiTool
from Tools.utilities_for_tools import load_refined_model, load_chain
import json
import fire
from Lang_file import Lanchain_impl
import os
import re
from langchain import PromptTemplate
from langchain import LLMChain
from utils import extract_values, read_json, write_answers, prepare_question_list
import datetime
import os
import logging
import torch, gc


log_dir = "./logs/"

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"logs_{current_time}.log")


logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def merge_step_updated(output, few_shot, langchain_call, model_name):
    ques = output["input"]
    wikipedia_ans, wikidata_ans = extract_values(output["output"])
    assistant_match = re.search(
        r"Assistant Response:\s*(.*)", output["output"], re.DOTALL
    )
    context = assistant_match.group(1).strip() if assistant_match else None
    int_knw = langchain_call.answer_ques(ques)
    print(f"context is ------> {context}")
    print(f"wikipedia answer is ------> {wikipedia_ans}")
    print(f"wikidata answer is  -------->{wikidata_ans}")
    print(f"internal knowledge answer is  -------->{int_knw}")
    template = """Your task is to provide short answers to questions. For doing this, you get answers that were extracted from Wikipedia, Wikidata and your own parametric knowledge respectively. You also get a paragraph of context information related to the answer of the question. 
                Only pick Internal Knowledge, if you have no answers either from Wikipedia nor Wikidata.
                Here are few examples to refer to.
                {example}
                Question: {ques}
                Wikipedia Answer: {wikipedia_ans}
                Wikidata Answer: {wikidata_ans}
                Internal Knowledge: {int_knw}
                Context: {context}
                Answer:


                """
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "ques",
            "context",
            "wikipedia_ans",
            "wikidata_ans",
            "int_knw",
            "example",
        ],
    )
    llm = load_chain(model_name)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    final_answer = llm_chain.run(
        {
            "ques": ques,
            "context": context,
            "wikipedia_ans": wikipedia_ans,
            "wikidata_ans": wikidata_ans,
            "int_knw": int_knw,
            "example": few_shot,
        }
    )
    return wikipedia_ans, wikidata_ans, int_knw, final_answer


#'gpt-3.5-turbo'
# gpt-4-0314
def main(
    dataset: str = "mintaka",
    model_name: str = "gpt-4-0314",
    output_path: str = "answers_data",
    dynamic=True,
):
    refined = load_refined_model()
    wiki_tool = WikiTool(model_name)
    path = os.getcwd()
    print("main---->", path)
    squall = Squall(
        f"{path}/Tools/Tools_Data/squall_fixed_few_shot.json", refined, model_name
    )
    sparql_tool = SparqlTool(f"{path}/Tools/Tools_Data/squall2sparql_revised.sh")
    questions = prepare_question_list(dataset)
    print(questions)
    print(refined)
    langchain_call = Lanchain_impl(
        dataset, model_name, wiki_tool, squall, sparql_tool, dynamic
    )
    final_answer_list = []
    for idx, question in enumerate(questions):
        temp = {}
        try:
            logging.info(
                f"----------Evaluation on Question: {question} Index: {idx}----------"
            )
            temp["question"] = question
            out, template_answer = langchain_call.execute_agent(question.strip("\n"))

            few_shot = read_json(dataset)
            wiki_ans, wikidata_ans, int_ans, final_answer = merge_step_updated(
                out, few_shot, langchain_call, model_name
            )
            temp["final_answer"] = final_answer.strip()
            temp["wikipedia_answer"] = wiki_ans
            temp["wikidata_answer"] = wikidata_ans
            temp["internal_knowledge"] = int_ans
            temp["error"] = None
            temp["intermediate_logs"] = template_answer
            final_answer_list.append(temp)
            logging.info(temp)
            logging.info(
                f"----Evaluation Done Question: {question} Index: {idx}---"
            )
        except Exception as e:
            # if "CUDA error:" in str(e):
            #     gc.collect()
            #     torch.cuda.empty_cache()
            temp["question"] = question
            temp["final_answer"] = None
            temp["intermediate_logs"] = None
            temp["error"] = str(e)
            final_answer_list.append(temp)
        del temp
        if (idx+1) % 10 == 0:
            write_answers(final_answer_list, output_path, dataset)
        continue

if __name__ == "__main__":
    fire.Fire(main)
