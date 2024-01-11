import os
import time
from enchant.utils import levenshtein
from langchain import PromptTemplate
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union, Dict, Any
from langchain.schema import AgentAction, AgentFinish
import re
from Tools.utilities_for_tools import load_sentence_transformer, load_chain
import json
import numpy as np
from sentence_transformers import util
import operator
from langchain.callbacks.manager import (
    Callbacks,
)
from langchain.callbacks import get_openai_callback
from langchain.agents import (
    Tool,
    AgentExecutor,
    AgentOutputParser,
    BaseSingleActionAgent,
)
from langchain.chat_models import ChatOpenAI
import traceback
import random
import backoff
import openai

import logging


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n\nObservation: {observation}\n\nThought: "
        # Set the agentHow many Hunger Games books were made into movies?_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class LLMSingleActionAgentCustom(BaseSingleActionAgent):
    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]

    @property
    def input_keys(self) -> List[str]:
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ):
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        user_question = kwargs.pop("input")
        return self.output_parser.parse(output, user_question)

    async def aplan(
        self,
        intermediate_steps,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

                Args:
                    intermediate_steps: Steps the LLM has taken to date,
                        along with observations
                    callbacks: Callbacks to run.
                    **kwargs: User inputs.
        Who was the longest-serving president in U.S. history? # longest-serving president in U.S. history
                Returns:
                    Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        user_question = kwargs.pop("input")
        return self.output_parser.parse(output, user_question)

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str, user_q) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        if "Wikipedia and Wikidata" in llm_output:
            a = re.search(r"\b(that is)\b", llm_output)
            if a is not None:
                llm_output = llm_output[a.end() :].strip()
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        if "Wikipedia_Answer" in llm_output and "Wikidata_Answer" in llm_output:
            wiki_regex = r"Wikipedia_Answer:(.*),"
            wikidata_regex = r"Wikidata_Answer:(.*)"
            wiki_answer = re.search(wiki_regex, llm_output, re.DOTALL)
            wikidata_answer = re.search(wikidata_regex, llm_output, re.DOTALL)
            wikipedia = wiki_answer.group(1).strip().strip("\n").strip(",")
            wikidata = wikidata_answer.group(1).strip().strip("\n")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={
                    "output": f"Wikipedia_Answer: {wikipedia}, Wikidata_Answer: {wikidata}"
                },
                log=llm_output,
            )

        # Parse out the action and action input
        regex_action = r"Action:(.*?)(?=\s*Action Input:|$)"
        regex_action_input = r"Action Input:(.*)"
        print(f"LLMOUtpUT----->{llm_output}")
        match_action = re.search(regex_action, llm_output, re.DOTALL)
        match_action_input = re.search(regex_action_input, llm_output, re.DOTALL)
        if match_action is None and match_action_input is None:
            print("Non returnable zone!")

            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={
                    "output": "Wikipedia Answer: None, Wikidata Answer: None "
                },
                log=llm_output,
            )
        print(f"Match action---->{match_action}")
        print(f"Match action input---->{match_action_input}")
        action = match_action.group(1).strip("\n").strip()
        action_input = match_action_input.group(1).strip("\n").strip()

        # Return the action and action input
        if (
            action == "WikiSearch"
            or action == "GenerateSparql"
            or action == "WikiSearchSummary"
        ):
            logging.info(f"Using the Tool -----> {action}")
            # print(f"Action is the Tool -----> {action}")
            # print(f"Action Input the Tool -----> {action_input}")
            return AgentAction(
                tool=action, tool_input=f"{user_q} # {action_input}", log=llm_output
            )
        else:
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class CustomOutputParserWikidata(AgentOutputParser):
    def parse(self, llm_output: str, user_q) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        if "Wikipedia and Wikidata" in llm_output:
            a = re.search(r"\b(that is)\b", llm_output)
            if a is not None:
                llm_output = llm_output[a.end() :].strip()
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        if "Wikipedia_Answer" in llm_output and "Wikidata_Answer" in llm_output:
            wiki_regex = r"Wikipedia_Answer:(.*),"
            wikidata_regex = r"Wikidata_Answer:(.*)"
            wiki_answer = re.search(wiki_regex, llm_output, re.DOTALL)
            wikidata_answer = re.search(wikidata_regex, llm_output, re.DOTALL)
            wikipedia = wiki_answer.group(1).strip().strip("\n").strip(",")
            wikidata = wikidata_answer.group(1).strip().strip("\n")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={
                    "output": f"Wikipedia_Answer: {wikipedia}, Wikidata_Answer: {wikidata}"
                },
                log=llm_output,
            )

        # Parse out the action and action input
        regex_action = r"Action:(.*?)(?=\s*Action Input:|$)"
        regex_action_input = r"Action Input:(.*)"
        print(f"LLMOUtpUT----->{llm_output}")
        match_action = re.search(regex_action, llm_output, re.DOTALL)
        match_action_input = re.search(regex_action_input, llm_output, re.DOTALL)
        if match_action is None and match_action_input is None:
            print("Non returnable zone!")

            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={
                    "output": "Wikipedia Answer: None, Wikidata Answer: None "
                },
                log=llm_output,
            )
        print(f"Match action---->{match_action}")
        print(f"Match action input---->{match_action_input}")
        action = match_action.group(1).strip("\n").strip()
        action_input = match_action_input.group(1).strip("\n").strip()

        # Return the action and action input
        if (
            action == "WikiSearch"
            or action == "GenerateSparql"
            or action == "WikiSearchSummary"
        ):
            logging.info(f"Using the Tool -----> {action}")
            # print(f"Action is the Tool -----> {action}")
            # print(f"Action Input the Tool -----> {action_input}")
            return AgentAction(
                tool=action, tool_input=f"{action_input}", log=llm_output
            )
        else:
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class Template_Construction:
    def __init__(self, question, dataset, wikidata=False, cos=False, div=False):
        self.model = load_sentence_transformer()
        self.question = question
        self.dataset = dataset
        self.wikidata = wikidata
        self.cos = cos
        self.div = div

    def cos_sim(self, element, model, labels_sim, threshold=2):
        x = model.encode([element])
        res = util.dot_score(x, labels_sim)
        res = res.squeeze()
        y = np.array(res)
        ind = np.argpartition(y, -threshold)[-threshold:]

        ind = ind[np.argsort(y[ind])]
        return ind

    def cos_sim_least(
        self, element, model, labels_sim, threshold=2, most_similar=False
    ):
        x = model.encode([element])
        res = util.dot_score(x, labels_sim)
        res = res.squeeze()
        y = np.array(res)
        if most_similar:
            ind = np.argpartition(y, -threshold)[-threshold:]

        else:
            ind = np.argpartition(y, threshold)[:threshold]
        ind = ind[np.argsort(y[ind])]
        return ind

    def most_similar_items(self, question, questions, threshold=1):
        labels_sim = self.model.encode(questions)
        indexes = self.cos_sim(question, self.model, labels_sim, threshold=threshold)
        if len(indexes) == 1:
            res_list = operator.itemgetter(*indexes)(questions)
        else:
            res_list = list(operator.itemgetter(*indexes)(questions))
        return res_list
    def DPP(self,q, final_list):
        K = q * final_list * q
        obj_log_det = LogDeterminantFunction(n=20,
                                            mode="dense",
                                            lambdaVal=0,
                                            sijs=K)
        greedy_indices_and_scores = obj_log_det.maximize(budget=3,
                                                            optimizer='NaiveGreedy',
                                                            stopIfZeroGain=False,
                                                            stopIfNegativeGain=False,
                                                            verbose=False)
        greedy_indices, greedy_scores = zip(*greedy_indices_and_scores)
        return greedy_indices
    def load_dataset_for_few_shot(self, path):
        questions = []
        with open(path, "r") as file:
            data = json.load(file)
        for x in data:
            if x.get("Question") is not None:
                questions.append(x.get("Question"))
        return questions
    def few_shot_with_dpp(self):
        path = os.getcwd()
        dataset ='compmix'
        questions = self.load_dataset_for_few_shot(f"{path}/data/{self.dataset}.json")
        q = self.most_similar_items(self.question, questions)
        with open(f"{path}/data/{dataset}.json", "r") as file:
            data = json.load(file)
            action_sequence = ""
            action_sequence_list = []
            final_template = ""
            for idx, x in enumerate(data):
                    action_sequence_list.append(x.get("Action_Sequence"))
                    final_list = []
                    for i in action_sequence_list:
                            x = []
                            for j in action_sequence_list:
                                    x.append(1 - levenshtein(i, j)/max(len(i),len(j)))
                            final_list.append(x)
                    final_list = np.array(final_list)
        indices = self.DPP(q, final_list)
        final_template = f"Example 1:{data[indices[0]].get('One_Shot')}\n\nExample 2:\n\n{data[indices[1]].get('One_Shot')}\n\nExample 3:\n\n{data[indices[2]].get('One_Shot')}"
        return final_template
    
    def full_shot_with_diversity(self):
        path = os.getcwd()
        questions = self.load_dataset_for_few_shot(f"{path}/data/{self.dataset}.json")
        fetching_ques = self.most_similar_items(self.question, questions)
        print(fetching_ques)
        with open(f"{path}/data/{self.dataset}.json", "r") as file:
            data = json.load(file)
            action_sequence = ""
            action_sequence_list = []
            final_template = ""
            for idx, x in enumerate(data):
                if x.get("Question").strip() == fetching_ques.strip():
                    action_sequence = x.get("Action_Sequence").strip().strip("\t")
                    final_template = (
                        f"Example 1: \n\n{final_template}{x.get('One_Shot')}"
                    )
                action_sequence_list.append(
                    x.get("Action_Sequence").strip().strip("\t")
                )
            similar_sequences = self.model.encode(action_sequence_list)
            if self.cos:
                print(f"doing cos max")
                indexes = self.cos_sim_least(
                    action_sequence, self.model, similar_sequences, 3, most_similar=True
                )
                final_template = f"Example 1:{final_template}\n\nExample 2:\n\n{data[indexes[1]].get('One_Shot')}\n\nExample 3:\n\n{data[indexes[2]].get('One_Shot')}"
            elif self.div:
                print(f"doing cos min")
                indexes = self.cos_sim_least(
                    action_sequence,
                    self.model,
                    similar_sequences,
                    10,
                    most_similar=False,
                )
                selected_indices = random.sample(list(indexes), 3)
                final_template = f"Example 1:{data[selected_indices[0]].get('One_Shot')}\n\nExample 2:\n\n{data[selected_indices[1]].get('One_Shot')}\n\nExample 3:\n\n{data[selected_indices[2]].get('One_Shot')}"
            else:
                print("cos general")
                indexes = self.cos_sim_least(
                    action_sequence, self.model, similar_sequences, 2
                )
                final_template = f"{final_template}\n\nExample 2:\n\n{data[indexes[0]].get('One_Shot')}\n\nExample 3:\n\n{data[indexes[1]].get('One_Shot')}"
            return final_template

    def static_prompt_construction(self):
        path = os.getcwd()
        if self.wikidata:
            questions = self.load_dataset_for_few_shot(
                f"{path}/data_late_fusion/{self.dataset}.json"
            )
            full_path = f"{path}/data_late_fusion/{self.dataset}.json"
        else:
            questions = self.load_dataset_for_few_shot(
                f"{path}/data/{self.dataset}.json"
            )
            full_path = f"{path}/data/{self.dataset}.json"
        # random.seed(4)
        selected_questions = random.sample(questions, 3)  # change static variable
        with open(full_path, "r") as file:
            data = json.load(file)
            final_template = ""
            counter = 0
            for x in data:
                for question in selected_questions:
                    if x.get("Question").strip() == question.strip():
                        counter = counter + 1
                        final_template = f"{final_template}\n\nExample {counter}: \n\n{x.get('One_Shot')}"

            return final_template


class Lanchain_impl:
    def __init__(self, dataset, model_name, wiki_tool, squall, sparql_tool, dynamic, DPP):
        self.dataset = dataset
        self.model_name = model_name
        self.wiki_tool = wiki_tool
        self.squall = squall
        self.sparql_tool = sparql_tool
        self.dynamic = dynamic
        self.DPP = DPP
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def exponential_backoff_get_wikipedia_summary_keyword(self, x):
        return self.wiki_tool.get_wikipedia_summary_keyword(x)

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def exponential_backoff_generate_squall_query(self, x):
        return self.squall.generate_squall_query(x)

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def exponential_backoff_get_wikipedia_summary(self, x):
        return self.wiki_tool.get_wikipedia_summary(x)

    def get_observation(self, x):
        print("Inside GetObservation!")
        if "Observation" in x and ":" in x:
            _, observation = x.split(":", 1)
            return observation
        else:
            return x

    def get_tools(self):
        tools = [
            Tool(
                name="WikiSearch",
                func=lambda x: self.exponential_backoff_get_wikipedia_summary_keyword(
                    x
                ),
                description="Useful to find relevant Wikipedia article given the Action Input. Do not use this tool with same Action Input.",
            ),
            Tool(
                name="GetWikidataID",
                func=lambda x: self.wiki_tool.all_wikidata_ids(x),
                description="useful to get QIDs given the Action Input. Do not use this tool with same Action Input.",
            ),
            Tool(
                name="GenerateSparql",
                func=lambda x: self.exponential_backoff_generate_squall_query(x),
                description="useful to get Squall query given the Action Input. Do not use this tool with same Action Input.",
            ),
            Tool(
                name="RunSparql",
                func=lambda x: self.sparql_tool.run_sparql(x),
                description="useful to run a query on wikibase to get results. Do not use this tool with same Action Input.",
            ),
            Tool(
                name="WikiSearchSummary",
                func=lambda x: self.exponential_backoff_get_wikipedia_summary(x),
                description="useful to find the answer on wikipedia article given the Action Input if WikiSearch Tool doesnt provide any answer!. Do not use this tool with same Action Input.",
                verbose=True,
            ),
            Tool(
                name="GetLabel",
                func=lambda x: self.wiki_tool.get_label(x),
                description="useful to get the label for the wikidata QID. Do not use this tool with same Action Input.",
            ),
        ]
        return tools

    def get_prompt(self, question, dynamic, DPP):
        workflow = ""
        if DPP:
            workflow = f"{workflow}{Template_Construction(question, self.dataset).few_shot_with_dpp()}"
        if dynamic:
            workflow = f"{workflow}{Template_Construction(question, self.dataset, div=True).full_shot_with_diversity()}"
        else:
            workflow = f"{workflow}{Template_Construction(question, self.dataset).static_prompt_construction()}"
        prepend_template = """Given the question, your task is to find the answer using both Wikipedia and Wikidata Databases.If you found the answer using Wikipedia Article you need to verify it with Wikidata, even if you do not find an answer with Wikpedia, first make sure to look up on different relevant wikipedia articles. If you still cannot find with wikipedia, try with Wikidata as well. 
When Wikipedia gives no answer or SPARQL query gives no result, you are allowed to use relevant keywords for finding QIDs to generate the SPARQL query.
Your immediate steps include finding relevant wikipedia articles summary to find the answer using {tools} provided, find Keywords that are the QIDS from the Wikidata using Wikipedia Page title. \nUse these QIDs to generate the SPARQL query using available {tools}.\nWikidata Answers are the observation after executing the SPARQL query.\n
Also do not check the wikipedia page manually to answer the questions. 
You have access to the following - {tools}!. 
Once you have the Wikipedia Answer and Wikidata Answer or either of them after trying, always follow the specific format to output the final answer - 
Final Answer: Wikipedia_Answer : Wikipedia Answer, Wikidata_Answer : Wikidata Answer , 
Assistant Response: Extended Answer that contains your reasoning, proof and final answer, please keep this descriptive.
Please do not use same Action Input to the tools, If no answer is found even after multiple tries using wikidata but found answer with wikipedia return and vice versa
Wikipedia_Answer : Answer, Wikidata_Answer : None 
Here are three examples to look at on how to use the {tools}\n
"""
        # Here are three examples to look at on how to use the {tools}\n
        additional_template = """
You have access to the following - {tools}!
Use the following format:
Question: the input question for which you must provide a natural language answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Always use the following format for the Final Answer -
Final Answer: Wikipedia_Answer : , Wikidata_Answer : ,
Assistant Response : 
Question: {input}
{agent_scratchpad}

            """

        workflow = workflow.strip("\n")
        complete_workflow = f"{prepend_template}{workflow}\n\n{additional_template.strip()}"  # {workflow}
        print(complete_workflow)
        logging.info(complete_workflow)
        prompt = CustomPromptTemplate(
            template=complete_workflow.strip("\n"),
            tools=self.get_tools(),
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"],
        )
        # print(prompt)
        return prompt

    def answer_ques(self, ques):
        template = """Given a question your task is to answer the question, please do not provide any other information other than the Answer
                    Question: {ques}
                    Answer: """
        prompt = PromptTemplate(template=template, input_variables=["ques"])
        llm = load_chain(self.model_name)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain.run(ques)

    def get_template_inference(self, out, int_answer):
        complete_steps = ""
        for i in out.get("intermediate_steps"):
            thought = ""
            if "Thought:" in i[0].log:
                complete_steps = (
                    f"{complete_steps}\n{thought}{i[0].log}\nObservation:{i[1]}\n"
                )
            else:
                thought = "Thought: "
                complete_steps = (
                    f"{complete_steps}\n{thought}{i[0].log}\nObservation:{i[1]}\n"
                )
        final_string = """Thought: I now know the final answer based on both Wikipedia and Wikidata. \nFinal Answer: """
        complete_steps = f'{out.get("input")}\n{complete_steps}\n{final_string}{out.get("output")}, Internal Knowledge: {int_answer}'
        return complete_steps

    def execute_one_query(self, complete_steps, x):
        temp = {}
        temp["question"] = x
        parametric_knowledge = self.answer_ques(x.strip())
        temp["internal_knowledge"] = parametric_knowledge.strip()
        try:
            answers = complete_steps  # agent_executor.run(x.strip())
            idx = answers.find("Wikidata_answer")
            _, wiki_answer = answers[:idx].split(":", 1)
            _, wikidata_answer = answers[idx:].split(":", 1)
            temp["wikipedia"] = (
                wiki_answer.replace(",", "").strip()
                if "None" not in wiki_answer
                else None
            )
            temp["wikidata"] = (
                wikidata_answer.strip() if "None" not in wikidata_answer else None
            )
            if "Wikidata_answer" not in answers and "Wikipedia_answer" not in answers:
                temp["error"] = answers
            else:
                temp["error"] = None
        except Exception as e:
            temp["wikipedia"] = None
            temp["wikidata"] = None
            temp["error"] = str(e)
            temp["stack_trace"] = str(traceback.print_exc())
        return temp

    def execute_agent(self, question):
        llm = ChatOpenAI(model_name=self.model_name, temperature=0, request_timeout=300)
        prompt = self.get_prompt(question, self.dynamic)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tools = self.get_tools()
        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        agent = LLMSingleActionAgentCustom(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        internal_answer = self.answer_ques(question)
        with get_openai_callback() as cb:
            out = agent_executor(question)
            logging.info(f"Total Tokens: {cb.total_tokens}")
            logging.info(f"Prompt Tokens: {cb.prompt_tokens}")
            logging.info(f"Completion Tokens: {cb.completion_tokens}")
            logging.info(f"Total Cost (USD): ${cb.total_cost}")
            answer_template_for_inference = self.get_template_inference(
                out, internal_answer
            )
        return out, answer_template_for_inference, cb.completion_tokens


# class Lanchain_impl_wikidata:
#     def __init__(self, dataset, model_name, wiki_tool, squall, sparql_tool, dynamic):
#         self.dataset = dataset
#         self.model_name = model_name
#         self.squall = squall
#         self.sparql_tool = sparql_tool
#         self.wiki_tool = wiki_tool
#         self.dynamic = dynamic

#     @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
#     def exponential_backoff_generate_squall_query(self, x):
#         return self.squall.generate_squall_query(x)

#     def get_observation(self, x):
#         print("Inside GetObservation!")
#         if "Observation" in x and ":" in x:
#             _, observation = x.split(":", 1)
#             return observation
#         else:
#             return x

#     def get_tools(self):
#         tools = [
#             Tool(
#                 name="GenerateSparql",
#                 func=lambda x: self.exponential_backoff_generate_squall_query(x),
#                 description="useful to get Squall query given the Action Input. Do not use this tool with same Action Input.",
#             ),
#             Tool(
#                 name="RunSparql",
#                 func=lambda x: self.sparql_tool.run_sparql(x),
#                 description="useful to run a query on wikibase to get results. Do not use this tool with same Action Input.",
#             ),
#             Tool(
#                 name="GetLabel",
#                 func=lambda x: self.wiki_tool.get_label(x),
#                 description="useful to get the label for the wikidata QID. Do not use this tool with same Action Input.",
#             ),
#         ]
#         return tools

#     def get_prompt(self, question, dynamic):
#         workflow = ""
#         if dynamic:
#             workflow = f"{workflow}{Template_Construction(question, self.dataset).full_shot_with_diversity()}"
#         else:
#             workflow = f"{workflow}{Template_Construction(question, self.dataset, wikidata= True).static_prompt_construction()}"
#         prepend_template = """Given the question, your task is to find the answer using Wikidata Databases.You are not allowed to use your own knowledge to get QIDs to generate the SPARQL query.
# Your immediate steps include generating SPARQL queries using available {tools} and then executing the query over Wikidata.\nWikidata Answers are the observation after executing the SPARQL query.\n
# You have access to the following - {tools}!.
# Once you have the Wikidata Answer, always follow the specific format to output the final answer -
# Final Answer: Wikidata_Answer : Wikidata Answer ,
# Assistant Response: Extended Answer that contains your reasoning, proof and final answer, please keep this descriptive.
# Please do not use same Action Input to the tools, If no answer is found even after multiple tries using wikidata
# Wikidata_Answer : None ,"""

#         # Here is one example to look at on how to use the {tools}\n"""
#         additional_template = """
# You have access to the following - {tools}!
# Use the following format:
# Question: the input question for which you must provide a natural language answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Always use the following format for the Final Answer -
# Final Answer: Wikidata_Answer : ,
# Assistant Response :
# Question: {input}
# {agent_scratchpad}

#             """

#         workflow = workflow.strip("\n")
#         complete_workflow = (
#             f"{prepend_template}{workflow}\n\n{additional_template.strip()}"  # {workflow}
#         )
#         logging.info(complete_workflow)
#         print(complete_workflow)
#         prompt = CustomPromptTemplate(
#             template=complete_workflow.strip("\n"),
#             tools=self.get_tools(),
#             # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
#             # This includes the `intermediate_steps` variable because that is needed
#             input_variables=["input", "intermediate_steps"],
#         )

#         return prompt

#     def answer_ques(self, ques):
#         template = """Given a question your task is to answer the question, please do not provide any other information other than the Answer
#                     Question: {ques}
#                     Answer: """
#         prompt = PromptTemplate(template=template, input_variables=["ques"])
#         llm = load_chain(self.model_name)
#         llm_chain = LLMChain(prompt=prompt, llm=llm)
#         return llm_chain.run(ques)

#     def get_template_inference(self, out, int_answer):
#         complete_steps = ""
#         for i in out.get("intermediate_steps"):
#             thought = ""
#             if "Thought:" in i[0].log:
#                 complete_steps = (
#                     f"{complete_steps}\n{thought}{i[0].log}\nObservation:{i[1]}\n"
#                 )
#             else:
#                 thought = "Thought: "
#                 complete_steps = (
#                     f"{complete_steps}\n{thought}{i[0].log}\nObservation:{i[1]}\n"
#                 )
#         final_string = """Thought: I now know the final answer based on  Wikidata. \nFinal Answer: """
#         complete_steps = f'{out.get("input")}\n{complete_steps}\n{final_string}{out.get("output")}, Internal Knowledge: {int_answer}'
#         return complete_steps

#     def execute_one_query(self, complete_steps, x):
#         temp = {}
#         temp["question"] = x
#         parametric_knowledge = self.answer_ques(x.strip())
#         temp["internal_knowledge"] = parametric_knowledge.strip()
#         try:
#             answers = complete_steps  # agent_executor.run(x.strip())
#             idx = answers.find("Wikidata_answer")
#             _, wiki_answer = answers[:idx].split(":", 1)
#             _, wikidata_answer = answers[idx:].split(":", 1)
#             # temp["wikipedia"] = (
#             #     wiki_answer.replace(",", "").strip()
#             #     if "None" not in wiki_answer
#             #     else None
#             # )
#             temp["wikidata"] = (
#                 wikidata_answer.strip() if "None" not in wikidata_answer else None
#             )
#             if "Wikidata_answer" not in answers and "Wikipedia_answer" not in answers:
#                 temp["error"] = answers
#             else:
#                 temp["error"] = None
#         except Exception as e:
#             temp["wikipedia"] = None
#             temp["wikidata"] = None
#             temp["error"] = str(e)
#             temp["stack_trace"] = str(traceback.print_exc())
#         return temp

#     def execute_agent(self, question):
#         llm = ChatOpenAI(model_name=self.model_name, temperature=0, request_timeout=300)
#         prompt = self.get_prompt(question, self.dynamic)
#         llm_chain = LLMChain(llm=llm, prompt=prompt)
#         tools = self.get_tools()
#         tool_names = [tool.name for tool in tools]
#         output_parser = CustomOutputParserWikidata()
#         agent = LLMSingleActionAgentCustom(
#             llm_chain=llm_chain,
#             output_parser=output_parser,
#             stop=["\nObservation:"],
#             allowed_tools=tool_names,
#         )
#         agent_executor = AgentExecutor.from_agent_and_tools(
#             agent=agent,
#             tools=tools,
#             verbose=True,
#             handle_parsing_errors=True,
#             return_intermediate_steps=True,
#         )
#         internal_answer = self.answer_ques(question)
#         out = agent_executor(question)
#         answer_template_for_inference = self.get_template_inference(
#             out, internal_answer
#         )
#         return out, answer_template_for_inference
