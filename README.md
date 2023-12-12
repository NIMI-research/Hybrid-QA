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
```

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

-----------HPC Setup TUD Barnard----------

🚩 Note: If you set everything up already, start from Step **4** with every new nession.


1. Create a workspace
```bash
# on Barnard
ws_allocate -F horse hybridQA 90

#on Alpha (A100 GPU Cluster)
ws_allocate -F beegfs hybridQA 90
```
2. **Clone the repository**:
```bash
git clone https://github.com/dhananjaybhandiwad/HybridQA.git
```
3. Setup the code including the creation of the virtual env
```bash
bash hpc_setup.sh
```
4. Allocate Resources, Load Modules and activate virtual env (if not already done)

**Do this before every session!**
```bash
#cpu only run, interactive
srun -n 1 -c 32 -t 02:00:00 --mem-per-cpu 1972 --pty bash
source activate_env.sh
```
OR
```bash
#gpu run, interactive
srun -N 1 -n1 -c 48 -t 02:00:00 --hint=nomultithread --mem-per-cpu 10320 --gres=gpu:4 --pty bash
source activate_env.sh
```
5. Set Huggingface to your Workspace (Default is in your $HOME which has probably not enough disc space)
```bash
export HF_HOME=path/to/your/workspace/.cache

python main.py --dataset qald --model-name 'mosaicml/mpt-7b-8k' --refined_cache_dir $HF_HOME
```
