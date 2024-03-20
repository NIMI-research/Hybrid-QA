# HybridQA

## Getting Started

These instructions will help you set up the HybridQA project on your local machine.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dhananjaybhandiwad/HybridQA.git
   ```
2. Ensure you have a Python Version >=3.9
3. Create a bew virtual Env 
   ```bash
   python3.10 -m venv <env_name>
   ```
4. Activate your environment 
  ```bash
  source <env_name>/bin/activate
  ```
5. Install requirements
```bash
pip install -r requirements.txt
```
6. Go the Project folder
```bash
cd app
```
7. To run
```bash
python main.py --dataset mintaka --model_name gpt-4-0314 --dynamic True --DPP=False #If not Dynamic, runs Static pipeline 
```
8. To run Wikipedia pipeline 
```bash
python wikipedia_pipeline.py --dataset mintaka --model_name gpt-4-0314
```
9. To run Wikidata pipeline 
```bash
python wikidata_pipeline.py --dataset mintaka --model_name gpt-4-0314
```