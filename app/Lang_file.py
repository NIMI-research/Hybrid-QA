import os

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
from langchain.agents import Tool, AgentExecutor, AgentOutputParser, BaseSingleActionAgent
from langchain.chat_models import ChatOpenAI
import traceback



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
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agentHow many Hunger Games books were made into movies?_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
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

            a = re.search(r'\b(that is)\b', llm_output)
            if a is not None:
                llm_output = llm_output[a.end():].strip()
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*)\nAction Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentAction(tool="GetObservation", tool_input=llm_output, log=llm_output)
            # return AgentFinish(
            #     # Return values is generally always a dictionary with a single `output` key
            #     # It is not recommended to try anything else at the moment :)s
            #     return_values={"output": llm_output},
            #     log=llm_output,
            # )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        if action == "WikiSearch" or action == "GenerateSparql" or action == "WikiSearchSummary":
            return AgentAction(tool=action, tool_input=f"{user_q} # {action_input}", log=llm_output)
        # if action == "WikiSearchSummary":
        #     return AgentAction(tool=action, tool_input=f"{user_q} # {action_input}", log=llm_output)
        # if action == "GenerateSquall":
        #     return AgentAction(tool=action, tool_input=f"{user_q} # {action_input}", log=llm_output)
        else:
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class Template_Construction():
    def __init__(self, question, dataset):
        self.model = load_sentence_transformer()
        self.question = question
        self.dataset = dataset

    def cos_sim(self, element, model, labels_sim, threshold = 2):
        x = model.encode([element])
        res = (util.dot_score(x, labels_sim))
        res = res.squeeze()
        y = np.array(res)
        ind = np.argpartition(y, -threshold)[-threshold:]
        ind = ind[np.argsort(y[ind])]
        return ind
    def cos_sim_least(self, element, model, labels_sim, threshold=2):
        x = model.encode([element])
        res = (util.dot_score(x, labels_sim))
        res = res.squeeze()
        y = np.array(res)
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

    def load_dataset_for_few_shot(self, path):
        questions = []
        with open(path, "r") as file:
            data = json.load(file)
        for x in data:
            if x.get("Question") is not None:
                questions.append(x.get("Question"))
        return questions

    def full_shot_with_diversity(self):
        path = os.getcwd()
        questions = self.load_dataset_for_few_shot(f"{path}/data/{self.dataset}.json")
        fetching_ques = self.most_similar_items(self.question, questions)
        with open(f"{path}/data/{self.dataset}.json", "r") as file:
            data = json.load(file)
            action_sequence = ""
            action_sequence_list = []
            final_template = ""
            for idx, x in enumerate(data):
                if x.get("Question").strip() == fetching_ques.strip():
                    action_sequence = x.get("Action_Sequence").strip().strip("\t")
                    final_template = f"Example 1: \n\n{final_template}{x.get('One_Shot')}"
                action_sequence_list.append(x.get("Action_Sequence").strip().strip("\t"))
            similar_sequences = self.model.encode(action_sequence_list)
            indexes = self.cos_sim_least(action_sequence, self.model, similar_sequences, 2)
            final_template = f"{final_template}\n\nExample 2:\n\n{data[indexes[0]].get('One_Shot')}\n\nExample 3:\n\n{data[indexes[1]].get('One_Shot')}"
            return final_template



class Lanchain_impl():
    def __init__(self, dataset, model_name, wiki_tool, squall, sparql_tool):
        self.dataset = dataset
        self.model_name = model_name
        self.wiki_tool = wiki_tool
        self.squall = squall
        self.sparql_tool = sparql_tool

    def get_observation(self,x):
        if "Observation" in x and ":" in x:
            _, observation = x.split(":", 1)
            return observation
        else:
            return x

    def get_tools(self):
        tools = [Tool(
                name="WikiSearch",
                func=lambda x: self.wiki_tool.get_wikipedia_summary_keyword(x),
                description="useful to find relevant wikipedia article given the Action Input"
            ),
            Tool(
                name="GetWikidataID",
                func=lambda x: self.wiki_tool.all_wikidata_ids(x),
                description="useful to get QIDs given the Action Input"
            ),
            Tool(
                name="GenerateSparql",
                func=lambda x: self.squall.generate_squall_query(x),
                description="useful to get Squall query given the Action Input"
            ),
            Tool(
                name="RunSparql",
                func=lambda x: self.sparql_tool.run_sparql(x),
                description="useful to run a query on wikibase to get results"
            ),
            Tool(
                name="WikiSearchSummary",
                func=lambda x: self.wiki_tool.get_wikipedia_summary(x),
                description="useful to find the answer on wikipedia article given the Action Input if WikiSearch Tool doesnt provide any answer!",
                verbose=True,
            ),
            Tool(
                name='GetLabel',
                func=lambda x: self.wiki_tool.get_label(x),
                description="useful to get the label for the wikidata QID"),
            Tool(
                name='GetObservation',
                func=lambda x: self.get_observation(x),
                description="useful to get the Observation for the LLM")
        ]
        return tools




    def get_prompt(self,question):
        workflow = Template_Construction(question, self.dataset).full_shot_with_diversity()
        prepend_template = """Given the question, your task is to find the answer using both Wikipedia and Wikidata Databases.If you found the answer using Wikipedia Article you need to verify it with Wikidata, even if you do not find an answer with Wikpedia, first make sure to look up on different relevant wikipedia articles. If you still cannot find with wikipedia, try with Wikidata as well. 
When Wikipedia gives no answer or SPARQL query gives no result, you are allowed to use relevant keywords for finding QIDs to generate the SPARQL query.
Your immediate steps include finding relevant wikipedia articles summary to find the answer, find Keywords that are the QIDS from the Wikidata using Wikipedia Page title. \nUse these QIDs to generate the SPARQL query using available {tools}.\nWikidata Answers are the observation after executing the SPARQL query.\n
Always follow the specific format to output the answer - 
Wikipedia_answer : Wikipedia Answer, Wikidata_answer : Wikidata Answer , Assistance Response: Extended Answer that containing your reasoning, proof and final answer, please keep this descriptive.
if no answer is found using wikidata but found answer with wikipedia return 
Wikipedia_answer : Answer, Wikidata_answer : None , Assistance Response: And extended Answer containing your reasoning and proof, please keep this descriptive.

Here are three examples to look at\n"""
        additional_template = """
Use the following format:
Question: the input question for which you must provide a natural language answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer...
Final Answer: Wikipedia_answer : , Wikidata_answer : ,
Assistance Response : 
        
Question: {input}
{agent_scratchpad}
                    
        """

        workflow=workflow.strip("\n")
        complete_workflow = f"{prepend_template}{workflow}\n\n{additional_template.strip()}"
        print(complete_workflow)
        prompt = CustomPromptTemplate(
            template = complete_workflow.strip("\n"),
            tools= self.get_tools(),
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
        
        return prompt

    def answer_ques(self, ques):
        template = """Given a question your task is to answer the question, please do not provide any other information other than the Answer
                    Question: {ques}
                    Answer: """
        prompt = PromptTemplate(
            template=template,
            input_variables=['ques'])
        llm = load_chain(self.model_name)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain.run(ques)

    def get_template_inference(self, out, int_answer):
        complete_steps = ""
        for i in out.get("intermediate_steps"):
            thought = ""
            if "Thought:" in i[0].log:
                complete_steps = f"{complete_steps}\n{thought}{i[0].log}\nObservation:{i[1]}\n"
            else:
                thought = "Thought: "
                complete_steps = f"{complete_steps}\n{thought}{i[0].log}\nObservation:{i[1]}\n"
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
            temp["wikipedia"] = wiki_answer.replace(",", "").strip() if "None" not in wiki_answer else None
            temp["wikidata"] = wikidata_answer.strip() if "None" not in wikidata_answer else None
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
        llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        prompt = self.get_prompt(question)
        llm_chain = LLMChain(llm=llm, prompt = prompt)
        tools = self.get_tools()
        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        agent = LLMSingleActionAgentCustom(
            llm_chain = llm_chain,
            output_parser = output_parser,
            stop = ["\nObservation:"],
            allowed_tools = tool_names,
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,

                                                         handle_parsing_errors=True, return_intermediate_steps=True)
        internal_answer = self.answer_ques(question)
        out = agent_executor(question)
        answer_template_for_inference = self.get_template_inference(out,internal_answer)
        print(answer_template_for_inference)
        return out, answer_template_for_inference
