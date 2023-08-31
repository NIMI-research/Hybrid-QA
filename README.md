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
python main.py --dataset qald #for other datasets replace qald with mintaka or compmix


-----------Using Conda----------

Make sure you are in the base env before creating the Conda env
1. Create a bew Conda env
   ```bash
   conda create --name hybrid python==3.10 ##env name is hybrid
   ```
3. Activate your environment 
  ```bash
  conda activate hybrid
  ```
3. Install requirements
```bash
pip install -r requirements.txt
```
4. Go to the project directory
```bash
cd app
```
5. To run the application
```bash
python main.py --dataset qald
```
