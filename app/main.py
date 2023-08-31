from Tools.Tool import Squall, SparqlTool, WikiTool
from Tools.utilities_for_tools import load_refined_model,load_chain
import json
import fire
from Lang_file import Lanchain_impl
import os
import re
from langchain import PromptTemplate
from langchain import LLMChain

def prepare_question_list(data_file):
    with open(f"data/{data_file}.txt", "r") as file:
        lines = file.readlines()
        return lines

def write_answers(answer_list, output_path, dataset, answer=True):
    json_write = json.dumps(answer_list, indent=4)

    # Writing to sample.json
    path = ""
    if answer:
        path = f"{output_path}/{dataset}_answer.json"
    else:
        path = f"{output_path}/{dataset}_templates.json"
    with open(path, "w") as outfile:
        outfile.write(json_write)

def read_json(dataset):
    path = os.getcwd()
    with open(f"{path}/data/merge_few_shot_examples.json","r") as file:
        data = json.load(file)
        return data[dataset]

def merge_step_updated(output, few_shot,langchain_call,model_name):
    ques = output['input']
    wikipedia_match = re.search(r'Wikipedia_Answer:\s*(\d+)', output['output'])
    wikidata_match = re.search(r'Wikidata_Answer:\s*(\d+)', output['output'])
    assistant_match = re.search(r'Assistant Response:\s*(.*)', output['output'], re.DOTALL)
    wikipedia_ans = wikipedia_match.group(1) if wikipedia_match else None
    wikidata_ans = wikidata_match.group(1) if wikidata_match else None
    context = assistant_match.group(1).strip() if assistant_match else None
    int_knw = langchain_call.answer_ques(ques)

    template = """Your task is to provide short answers to questions. For doing this, you get answers that were extracted from Wikipedia, Wikidata and your own parametric knowledge respectively. You also get a paragraph of context information related to the answer of the question. 
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
        input_variables=['ques', 'context', 'wikipedia_ans', 'wikidata_ans', 'int_knw', 'example'])
    llm = load_chain(model_name)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(
        {'ques': ques, 'context': context, 'wikipedia_ans': wikipedia_ans, 'wikidata_ans': wikidata_ans,
         'int_knw': int_knw, 'example': few_shot})


def main(dataset: str = "mintaka",
         model_name: str = "gpt-4-0314",
         output_path: str = "answers_data"
):
    refined = load_refined_model()
    wiki_tool = WikiTool(model_name)
    path = os.getcwd()
    print("main---->", path)
    squall = Squall(f"{path}/Tools/Tools_Data/squall_fixed_few_shot.json", refined,model_name)
    sparql_tool = SparqlTool(f"{path}/Tools/Tools_Data/squall2sparql_revised.sh")
    questions = prepare_question_list(dataset)
    print(questions)
    print(refined)
    langchain_call = Lanchain_impl(dataset, model_name, wiki_tool, squall, sparql_tool)
    final_answer_list = []
    for question in questions:
        out, template_answer = langchain_call.execute_agent(question.strip("\n"))
        #answer_list.append(out)
        #template_list.append(template_answer)
        few_shot = read_json(dataset)
        final_answer = merge_step_updated(out,few_shot,langchain_call,model_name)
        final_answer_list.append({"question":question,"final_answer":final_answer})
    write_answers(final_answer_list, output_path, dataset)

if __name__ == "__main__":
    fire.Fire(main)
