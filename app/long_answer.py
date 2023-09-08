import os
import json
import re
from utils import write_answers
import fire
from langchain import PromptTemplate
from langchain import LLMChain
from Tools.utilities_for_tools import load_chain, load_openai_api


def long_answer_call(ques, wikipedia_ans, wikidata_ans, int_knw, context):
    template = """Your task is to answer questions. For doing this, you get short answers that were extracted from Wikipedia, Wikidata and your own parametric knowledge respectively. You also get a paragraph of context information related to the answer of the question. Do not use any information outside of those sources to answer the question unless neither Wikipedia nor Wikidata provide information. Given all of this information, you need to do your best to formulate an answer to the question, which is of high quality, truthful, informative and engaging.
Here are few examples to refer to.

Question: When did aerosmith form?
Wikipedia Answer: 1970
Wikidata Answer: 1969
Internal Knowledge: 1970
Context: Aerosmith is an American rock band that has sold more than 150 million records worldwide. According to the Wikipedia Article, Aerosmith was formed in the year 1970 in Boston. While the Wikidata answer is 1969, which could be the ‘inception’ and the ‘work period (start)’ is 1970. Wikipedia might be more reliable given the context and 1970 is the final answer 
Answer: Aerosmith is an American rock band that formed in 1970 according to Wikipedia. They sold more than 150 million records worldwide.

Question: How large is the area of uk?
Wikipedia Answer: 93,628 square miles (242,495 km2)
Wikidata Answer: 242495
Internal Knowledge: 242,495 square kilometres
Context: According to the Wikipedia article, the area of the United Kingdom is 93,628 square miles (242,495 km2). However the Wikidata gives the 242495 with no specific units, but it can be inferred from the Wikipedia answer that 242495 is km2 and 242,495 km2 is the final answer.
Answer: According to Wikipedia and Wikidata the area of the United Kingdom is 242,495 square kilometres. 

Question: What was the narrative location of the book wuthering heights?
Wikipedia Answer: West Yorkshire moors
Wikidata Answer: Yorkshire
Internal Knowledge: Yorkshire Moors, England
Context: According to the Wikipedia article, Wuthering Heights is an 1847 novel by Emily Brontë, set on the West Yorkshire moors in Northern England. The entity page of  Wuthering Heights returns Yorkshire
Answer: The Wuthering Heights novel by Emily Brontë was set on the West Yorkshire moors in Northern England. 

Question: Was harrison ford in star wars?
Wikipedia Answer: Yes
Wikidata Answer: - None
Internal Knowledge: Yes
Context: Star Wars is a media franchise consisting of nine films in the Skywalker Saga, including the original, prequel, and sequel trilogies. Harrison Ford starred as Han Solo in the original trilogy films alongside Mark Hamill as Luke Skywalker and Carrie Fisher as Leia Organa. Ford and other original trilogy cast members reprised their roles in the sequel trilogy films.
Answer: Harrison Ford starred as Han Solo in the original trilogy films. Ford and other original trilogy cast members reprised their roles in the sequel trilogy films

Question: What was the name of the coldplay album that was nominated for album of the year but did not win in 2021?
Wikipedia Answer: - None
Wikidata Answer: - None
Internal Knowledge: Everyday Life
Context: I could not find any information about the name of the Coldplay album that was nominated for album of the year but did not win in 2021 using both Wikipedia and Wikidata. However, using my own Internal knowledge, Everyday Life was the name of the coldplay album. 
Answer: I am not certain of the answer as I could not verify it in my knowledge sources, but I believe the Coldplay album “Everyday Life” was nominated for “Album of the Year” at the 2021 Grammy Awards and did not win. 

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
        ],
    )
    llm = load_chain("gpt-4-0314")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    long_answer = llm_chain.run(
        {
            "ques": ques,
            "context": context,
            "wikipedia_ans": wikipedia_ans,
            "wikidata_ans": wikidata_ans,
            "int_knw": int_knw,
        }
    )
    return long_answer


def main(
    input_json_file: str = "",
    output_path: str = "long_data/",
    dataset: str = "mintaka",
):
    paths = os.getcwd()
    print(paths)
    assist_list = []
    regex = r"Assistant Response:(.*)Internal Knowledge:(.*)"
    regex_again = r"Assistance Response:(.*)Internal Knowledge:(.*)"
    _ = load_openai_api()
    with open(f"{paths}/answers_data/{input_json_file}", "r") as file:
        data = json.load(file)
        print(len(data))
        for idx, each in enumerate(data):
            temp = {}
            if each.get("intermediate_logs") is not None:
                match = re.search(regex, each.get("intermediate_logs"), re.DOTALL)
                match_again = re.search(
                    regex_again, each.get("intermediate_logs"), re.DOTALL
                )
                temp["question"] = each.get("question")
                temp["internal_knowledge"] = each.get("internal_knowledge")
                temp["wikipedia_answer"] = each.get("wikipedia_answer")
                if each.get("wikidata_answer") is None:
                    temp["wikidata_answer"] = None
                elif "Assistant Response" in each.get(
                    "wikidata_answer"
                ) or "Assistance Response" in each.get("wikidata_answer"):
                    one, _ = each.get("wikidata_answer").split("\n", 1)
                    temp["wikidata_answer"] = one.strip()
                else:
                    temp["wikidata_answer"] = each.get("wikidata_answer")
                if match is not None:
                    temp["assistant_response"] = match.group(1).strip().strip(",")
                elif match_again is not None:
                    temp["assistant_response"] = match_again.group(1).strip().strip(",")
                else:
                    temp["assistant_response"] = None

                temp["long_answer"] = long_answer_call(
                    each.get("question"),
                    each.get("wikipedia_answer"),
                    each.get("wikidata_answer"),
                    each.get("internal_knowledge"),
                    temp["assistant_response"],
                )
            assist_list.append(temp)
            print(temp)
            del temp
        write_answers(assist_list, output_path, dataset)


if __name__ == "__main__":
    fire.Fire(main)
