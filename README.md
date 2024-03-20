# Beyond Boundaries: A Human-like Approach for Question Answering over Structured and Unstructured information sources
Answering factual questions from heterogeneous sources, such as graphs and text, is a key capacity of intelligent systems. Current approaches either (i) perform question answering over text and structured sources as separate pipelines followed by a merge step or (ii) provide an early integration, giving up the strengths of particular information sources. To solve this problem, we present "HumanIQ," a method that teaches language models to dynamically combine retrieved information by imitating how humans use retrieval tools. Our approach couples a generic method for gathering human demonstrations of tool use with adaptive few-shot learning for tool-augmented models. We show that HumanIQ confers significant benefits, including (i) reducing the error rate of our strongest baseline (GPT-4) by over 50% across 3 benchmarks, (ii) improving human preference over responses from vanilla GPT-4 (45.3% wins, 46.7% ties, 8.0% loss), and (iii) outperforming numerous task-specific baselines.
![pipeline](https://github.com/NIMI-research/HybridQA/assets/91888251/f28aa180-c98e-413f-825f-0dfe962683ea)

![bars](https://github.com/NIMI-research/HybridQA/assets/91888251/616497af-316a-42d1-beb5-1270d7d6f6de)

## Prerequisites

Before you begin, ensure you meet the following requirements:

 - Python version 3.9 or higher. The project is tested on Python 3.10 for optimal performance.
 - Virtual environment tool (e.g., venv, conda) for Python.
 - OpenAI API key
### Installation Guide

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NIMI-research/HybridQA.git
   ```
2. **Set Up a Python Virtual Environment**:
It's recommended to create a virtual environment to isolate project dependencies. Replace <env_name> with your preferred environment name.
   ```bash
   python3.10 -m venv <env_name>
   ```
1. **Activate the Virtual Environment**:
Activate the created virtual environment with the command below.
   ```bash
   source <env_name>/bin/activate
   ```
2. **Install Project Dependencies:**:
Navigate to the project directory and install the required Python packages.
   ```bash
   cd HybridQA
   pip install -r requirements.txt
   ```

### Usage
**Running the Main Pipeline**:
Use this command to run the primary pipeline. You can specify whether to run in dynamic or static or with Determinantal Point Process (DPP) for selecting the few shot examples.
```bash
cd app
python main.py --dataset mintaka --model_name gpt-4-0314 --dynamic True --DPP False
 ```
 --dataset: Mintaka or Qald or Compmix.
 --model_name: gpt-4-0314 or gpt-3.5-turbo or mpt30b
 --dynamic: Enables dynamic mode when set to True; use False for static mode.
 --DPP: Activates DPP for few-shot selection when True; False disables it.

**Wikipedia Pipeline**: 
```bash
python wikipedia_pipeline.py --dataset mintaka --model_name gpt-4-0314
 ```
**Wikipedia Pipeline**: 
```bash
python wikidata_pipeline.py --dataset mintaka --model_name gpt-4-0314
 ```
