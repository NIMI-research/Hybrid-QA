import numpy as np
import operator
import json
from sentence_transformers import SentenceTransformer, util
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem
from langchain import PromptTemplate
from langchain import OpenAI, LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
import re
from .utilities_for_tools import load_chain, load_openai_api
import openai
from .utilities_for_tools import load_sentence_transformer
import subprocess
from .Custom_Classes import CustomWikipediaAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
import requests
from typing import List, Dict, Any
import os
import logging


class Squall:
    def __init__(self, few_shot_path: str, refined, model_name, wikidata=False):
        self.few_shot_path = few_shot_path
        self.config = load_openai_api()
        self.refined = refined
        self.model = load_sentence_transformer()
        self.mode_name = "gpt-4-0314"
        self.wikidata = wikidata


    def cos_sim(self, element, model, labels_sim, threshold=2):
        x = model.encode([element])
        res = util.dot_score(x, labels_sim)
        res = res.squeeze()
        y = np.array(res)
        ind = np.argpartition(y, -threshold)[-threshold:]
        ind = ind[np.argsort(y[ind])]
        return ind

    def load_dataset(self):
        with open(self.few_shot_path, "r") as file:
            data_train = json.load(file)
            for i in data_train:
                if (
                    i["Question"]
                    == "What is the name of the sixth movie in the Harry Potter franchise?"
                ):
                    i[
                        "Completion"
                    ] = "Which <Q11424:film> is-<P179:part_of_the_series> <Q216930:Harry_Potter_film_series> at <P1545:series_ordinal> 6?"
                elif (
                    i["Question"] == "What was the name of the last Harry Potter movie?"
                ):
                    i[
                        "Completion"
                    ] = "Which <Q11424:film> is-<P179:part_of_the_series> <Q216930:Harry_Potter_film_series> has the latest <P577:publication_date>?"
            return data_train

    def create_question_squall_mapping(self):
        mapping = {}
        mapping_id = {}
        y = self.load_dataset()
        questions = []
        for x in y:
            if x.get("Question") is not None and x.get("Completion") is not None:
                mapping[x.get("Question").strip()] = x.get("Completion").strip()
                mapping[x.get("id")] = x.get("Question").strip()
                questions.append(x.get("Question").strip())
        return mapping, mapping_id, questions

    def most_similar_question(self, question: str):
        mapping, mapping_id, questions = self.create_question_squall_mapping()
        labels_sim = self.model.encode(questions)
        indexes = self.cos_sim(question, self.model, labels_sim)
        res_list = list(operator.itemgetter(*indexes)(questions))
        return res_list, mapping

    def get_label_and_description(self, entity_id):
        try:
            q42_dict = get_entity_dict_from_api(entity_id)
            q42 = WikidataItem(q42_dict)
            return q42.get_label(), q42.get_description()
        except Exception as e:
            return None, None

    def get_entity_linking_from_refined_batch(self, inputs: list) -> List[str]:
        threshold = 0.05
        docs = self.refined.process_text_batch(inputs)
        local_list = []
        for doc in docs:
            for span in doc.spans:
                if span.predicted_entity is not None:
                    qid = span.predicted_entity.wikidata_entity_id
                    if qid is not None:
                        label, desc = self.get_label_and_description(qid)
                        local_list.append((qid, label, desc))
        return local_list

    def get_llm_ent(self, ques):
        template = """Given the Question, your task is to only identify the possible Entities that have the QIDs from Wikibase in this question. Provide no other information
        Example: Question - Who ran for president in 2008 and was from Chicago, Illinois - Answer: President, Chicago, Illinois.
        Question: {ques}
        Answer: """
        prompt = PromptTemplate(template=template, input_variables=["ques"])
        llm = load_chain(self.mode_name)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        result = llm_chain.run(ques)
        return list(result.split(","))

    def perform_entity_disambiguation_davinci_003(
        self, question, ambiguation_dict
    ) -> List:
        final_list = []
        for k, v in ambiguation_dict.items():
            prompt = f"""Your are an expert in performing Entity Disambiguation task on wikidata IDs. Your task is to perform Entity 
                          Disambiguation given a Question and a list of wikidata IDs along with their description.
                          Question: {question}
                          wikidata IDs List: {v}
                          Perform the task given the above List.
                          Return the answer only from the above wikidata IDs List, dont try to answer the given Question I repeat dont try to answer the Question and dont provide any explanation.
                          Answer:"""
            response = openai.Completion.create(
                api_key=self.config["OPENAI"]["OPENAI_API_KEY"],
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=20,
                temperature=0,
                top_p=1,
                n=1,
                stop="/n",
            )
            regex = r"Q[0-9]*"
            llm_output = response["choices"][0]["text"].strip()
            match = re.search(regex, llm_output)
            if match is not None:
                label, desc = self.get_label_and_description(match.group(0))
                final_list.append((llm_output, label, desc))
        return final_list

    def get_entity_linking_from_refined(self, inputs: str):
        ambiguation_dict = {}
        threshold = 0.1
        question = inputs
        question = question.strip()
        index = 0
        single_entity_list = []
        spans = self.refined.process_text(question)
        for span in spans:
            predicted_entities = span.top_k_predicted_entities
            if predicted_entities is not None:
                predicted_span_lists = []
                for tuples in predicted_entities:
                    if (
                        "wikidata_entity_id" in str(tuples[0])
                        and float(tuples[1]) > threshold
                    ):
                        regex = r"Q[0-9]*"
                        match = re.search(regex, str(tuples[0]))
                        predicted_span_lists.append(match.group(0))
                if len(predicted_span_lists) == 1:
                    single_entity_list.extend(predicted_span_lists)
                if len(predicted_span_lists) > 1:
                    local_list = []
                    for qid in predicted_span_lists:
                        label, desc = self.get_label_and_description(qid)
                        if desc is not None:
                            local_list.append((qid, desc))
                    ambiguation_dict[index] = local_list
                    index = index + 1
                    del local_list
                del predicted_span_lists

        disambiguated_list = self.perform_entity_disambiguation_davinci_003(
            question, ambiguation_dict
        )
        entity_desc_list = []
        for qid in single_entity_list:
            label, desc = self.get_label_and_description(qid)
            if desc is not None:
                entity_desc_list.append((qid, label, desc))
        del single_entity_list
        if len(disambiguated_list) > 0:
            entity_desc_list.extend(disambiguated_list)
        return entity_desc_list, disambiguated_list

    def union_of_refined_entities(self, question):
        ents = self.get_llm_ent(question)
        for_question, disambiguation_list = self.get_entity_linking_from_refined(
            question
        )
        for_entities = self.get_entity_linking_from_refined_batch(ents)
        for dis in disambiguation_list:
            for ents in for_entities:
                if dis[1] == ents[1]:
                    for_entities.remove(ents)
        for_question.extend(for_entities)
        return list(set(for_question))

    def generate_prompt_based_on_similarity(self, question: str):
        most_similar_questions, mapping = self.most_similar_question(question)
        examples_list = []
        for x in most_similar_questions:
            example_dict = {}
            entities = self.union_of_refined_entities(x)
            example_dict["ques"] = x
            example_dict["entities"] = entities
            example_dict["answer"] = mapping.get(x)
            examples_list.append(example_dict)
            del example_dict
        return examples_list

    def generate_squall_query(self, actionInput: str):
        print("Inside GenerateSparql!")
        if self.wikidata:
            question = actionInput
            ent_list = self.union_of_refined_entities(question)
            entities = []
            for i in ent_list:
                entities.append(i[0])
        else:
            question, entities = actionInput.split("#")
            question, entities = question.strip(), entities.strip()
            entities = entities.replace("[", "").replace("]", "").split(",")
        entities = [e.strip().strip("'").strip('"') for e in entities]
        examples = self.generate_prompt_based_on_similarity(question)
        x = []
        regex = r"Q[0-9]*"
        for e in entities:
            match = re.search(regex, e)
            if e != "None" and match is not None:
                label, description = self.get_label_and_description(e.strip())
                x.append((e.strip(), label, description))
        entities = x
        few_shot_template = (
            """Question: {ques} \nEntities: {entities} \nSQUALL Query: {answer}"""
        )
        prefix = """Given the Question and Entities Generate the SQUALL Query. Here are the examples"""
        suffix = """Question: {ques} \nEntities: {entities} \nSQUALL Query:"""
        few_shot_prompt = PromptTemplate(
            input_variables=["ques", "entities", "answer"], template=few_shot_template
        )

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=few_shot_prompt,
            suffix=suffix,
            input_variables=["ques", "entities"],
        )
        llm = load_chain(self.mode_name)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        x = llm_chain.run({"ques": question, "entities": entities})
        path = os.getcwd()
        converter = SparqlTool(f"{path}/Tools/Tools_Data/squall2sparql_revised.sh")
        response = converter.gen_sparql_from_squall(x)
        return response


class SparqlTool:
    def __init__(self, path_to_conversion_tool: str):
        self.path = path_to_conversion_tool
        self.config = load_openai_api()

    def run_squall_tool(self, x: str):
        p = subprocess.Popen(
            [self.path, "-wikidata", x], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        if out.decode() == "":
            return "The above query is syntactically wrong please try with corrected syntax!"
        else:
            return out.decode().replace("\n", "").strip()

    def post_process_sparql(self, query: str):
        query = query.strip()
        update_query = []
        x = query.split(" ")
        for e in x:
            if "<" in e and ">" in e and ":" in e:
                if (
                    e.startswith("p")
                    or e.startswith("ps")
                    or e.startswith("pq")
                    or e.startswith("n1")
                    or e.startswith("n2")
                ):
                    ids, _ = e.split(":", 1)
                    e = ids.replace("<", ":").strip()
                elif (e.startswith("<") or e.endswith(">")) and ">" in e and "Q" in e:
                    ids, _ = e.split(":", 1)
                    e = ids.replace("<", "wd:").strip()
            update_query.append(e)
        return " ".join(update_query).strip()

    def gen_sparql_from_squall(self, query):
        y = self.run_squall_tool(query)
        message = "The possible reason is\n 1) The query is syntactically wrong\n"
        if (
            "The above query is syntactically wrong please try with corrected syntax!"
            not in y
        ):
            processed_sparql = self.post_process_sparql(y)
            return processed_sparql
        else:
            return message

    def get_nested_value(self, o: dict, path: list) -> any:
        current = o
        for key in path:
            try:
                current = current[key]
            except:
                return None
        return current

    def run_sparql(self, query: str, url="https://query.wikidata.org/sparql"):
        try:
            print("Inside RunSparql!")

            wikidata_user_agent_header = (
                None
                if not self.config.has_section("WIKIDATA")
                else self.config["WIKIDATA"]["WIKIDATA_USER_AGENT_HEADER"]
            )
            query = self.post_process_sparql(query)
            headers = {"Accept": "application/json"}
            if wikidata_user_agent_header is not None:
                headers["User-Agent"] = wikidata_user_agent_header
            response = requests.get(
                url, headers=headers, params={"query": query, "format": "json"}
            )
            if "boolean" in response.json():
                return {"message": response.json()["boolean"]}
            if response.status_code != 200:
                return "That query failed. Perhaps you could try a different one?"
            results = self.get_nested_value(response.json(), ["results", "bindings"])

            if len(results) == 0:
                return {
                    "message": """The given query failed, please reconstruct your query and try again."""
                }
            results_list = []
            x = results
            if len(x) > 0:
                for y in x:
                    if y.get("x1") is not None:
                        results_list.append({"value": y.get("x1").get("value")})
                    else:
                        results_list.append(y)
            return {"message": results_list}
        except Exception as e:
            return {
                "message": "The given query failed, please reconstruct your query and try again."
            }



class WikiTool:
    def __init__(self, model_name):
        self.config = load_openai_api()
        self.model_name = "gpt-4-0314"

    def get_label(self, entity_id):
        print("Inside GetLabel!")
        if "[" in entity_id and "]" in entity_id:
            entity_id = entity_id.replace("[", "").replace("]", "").strip()
            entity_id = entity_id.split(",")
        else:
            return "Try again by passing values in a Python List format with comma separated values!"
        results = []
        for e in entity_id:
            e = e.replace("'", "").replace('"', "")
            try:
                q42_dict = get_entity_dict_from_api(e.strip())
                q42 = WikidataItem(q42_dict)
                results.append(q42.get_label())
            except Exception as e:
                return "Most probable reason would be the entity label passed might be wrong!"
        return results

    def get_wikidata_id(self, page_title, language="en"):
        url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "format": "json",
            "titles": page_title,
        }

        response = requests.get(url, params=params)
        data = response.json()
        pages = data["query"]["pages"]
        for page in pages.values():
            x = page.get("pageprops", {}).get("wikibase_item")
            if x is not None:
                return x
            else:
                return "There is no QID for the given keyword, please retry with another relvant keyword for the QID from Wikidata pages."

    def all_wikidata_ids(self, actionInput):
        print("Inside GetWikidataId!")
        logging.info("Using the Tool -----> GetWikidataId")
        try:
            actionInput = actionInput.replace("[", "").replace("]", "")
            actionInput = actionInput.replace("'", "").replace('"', "").split(",")
            x = []
            for act in actionInput:
                x.append(self.get_wikidata_id(act.strip()))
            return x
        except Exception as e:
            return "There is an internal error while handling this request!"

    def get_wikipedia_summary(self, actionInput) -> str:
        print("Inside WikiSearchSummary!")
        logging.info("Using the Tool -----> WikiSearchSummary")
        ques, search = actionInput.split("#")
        ques, search = ques.strip(), search.replace("[", "").replace("]", "").strip()
        items = search.split(",")
        results = []
        wikipedia = WikipediaAPIWrapper(top_k_results=1)
        for item in items:
            results.append(wikipedia.run(item))
        results = "\n".join(results)
        template = """Question: {ques} with the provided context as {results}. show proof to the answer along with the Page Name along with summary.
                      If you dont find the answer in {results} Just say Answer not found in Context.
                      Answer: """
        prompt = PromptTemplate(template=template, input_variables=["ques", "results"])
        llm = load_chain(self.model_name)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return results

    def get_wikipedia_summary_keyword(self, actionInput) -> str:
        print("Inside WikiSearch!")
        ques, search = actionInput.split("#")
        ques, search = ques.strip(), search.strip()
        search = search.replace("'", "").replace('"', "")
        search = search.replace("[", "").replace("]", "").split(",")
        wikipedia_wrapper = CustomWikipediaAPIWrapper(top_k_results=3)
        result = wikipedia_wrapper.run(search)
        # wikipedia = WikipediaAPIWrapper(top_k_results=1)
        template = """Your task is to find only the relevant page out of all the given pages and not to find the page that answers the given Question.
                    Question: {search} with the provided context as {result} containing the Page title and the summary.Show only one relevant Page title that is relevant to the question, also summarize this page summary
                    Example here
                    Question: How many wives did Roman Emperor Vitellius have?
                    Context: ["Page: Vitellius\nSummary: Aulus Vitellius (; Latin: [ˈau̯lʊs wɪˈtɛlːijʊs]; 24 September 15 – 20 December 69) was Roman emperor for eight months, from 19 April to 20 December AD 69. Vitellius was proclaimed emperor following the quick succession of the previous emperors Galba and Otho, in a year of civil war known as the Year of the Four Emperors. Vitellius was the first to add the honorific cognomen Germanicus to his name instead of Caesar upon his accession. \n
                            Page: Year of the Four Emperors\nSummary: The Year of the Four Emperors, AD 69, was the first civil war of the Roman Empire, during which four emperors ruled in succession: Galba, Otho, Vitellius, and Vespasian. \n
                            Page: Otho\nSummary: Marcus Otho (; born Marcus Salvius Otho; 28 April 32 – 16 April 69) was the seventh Roman emperor, ruling for three months from 15 January to 16 April 69. He was the second emperor of the Year of the Four Emperors.]
                    Answer: Page: Vitellius\nSummary: Vitellius was a Roman Emperor who served from 16 April to 22 December in AD 69. He was the third and final emperor in the Year of the Four Emperors. He had two wives, Petronia, with whom he had a child, and Galeria Fundana, who bore him a son named Vitellius Germanicus
                    Question: {search}
                    Context: {result}
                    Answer:: """
        prompt = PromptTemplate(template=template, input_variables=["search", "result"])
        llm = load_chain("gpt-4-0314")
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain.run({"search": ques, "result": result})
