# VaxOpt: DQN-Based COVID-19 Vaccine Allocation

**VaxOpt** is a reinforcement learning framework for optimizing COVID-19 vaccine distribution across Indian states using Deep Q-Networks (DQN).

Inspired by the methodology in:
> **Awasthi et al. (2022)**, *Learning Effective Strategies for COVID-19 Vaccine Distribution using Reinforcement Learning.*  
> *Intelligence-Based Medicine*, 6, 100060.  
> [ScienceDirect Paper Link](https://doi.org/10.1016/j.ibmed.2022.100060)

VaxOpt provides a clean, lightweight PyTorch implementation designed for academic reproducibility, fast training, and Google Colab execution with automatic checkpoint saving and resume capabilities.

---

## 1. Methodology Summary

The vaccine allocation problem is formulated as an episodic reinforcement learning task:

- **State Space ($S \in \mathbb{R}^9$):** A continuous 9-dimensional context vector for each state on a given date:
  1. Confirmed cases ratio ($\frac{\text{Confirmed}_i}{\sum \text{Confirmed}}$)
  2. Predicted death rate ($dr$)
  3. Predicted recovery rate ($rr$)
  4. Population ratio ($\frac{\text{Pop}_i}{\sum \text{Pop}}$)
  5. Predicted susceptible ratio ($S_i = \frac{\text{Susceptible}_i}{\sum \text{Susceptible}}$)
  6. Overall hospital beds ratio
  7. Overall ICU beds ratio
  8. Ventilators ratio
  9. Age distribution ratio (>50 years)
- **Action Space ($\mathcal{A}$):** Discretized allocation buckets: $a \in \{0, 1, \dots, n-1\}$ where $n = 1000$ (default).
  - Allocation fraction: $A_i = \frac{a}{n}$.
  - Allocated doses: $V_i = A_i \times \text{Batch Size}$ ($\text{Batch Size} = 1,000,000$ doses).
- **Reward Function ($R_i$):**
  $$R_i = 100 \cdot \exp\left( - \frac{(A_i - S_i)^2}{10^{-4}} \right)$$
  Maximized when the allocation fraction $A_i$ closely matches the state's susceptible fraction $S_i$.
- **Environment Transition:**
  Administering $V_i$ doses decreases the state's susceptible count ($\text{Susceptible}_i' = \text{Susceptible}_i - V_i$), updating feature index 4. The episode terminates after 1 allocation step.
- **DQN Architecture:** Multi-Layer Perceptron (MLP) with 2 hidden layers of 64 units each ($9 \to 64 \to 64 \to 1000$) with ReLU activations and Adam optimizer ($\text{lr} = 10^{-3}$).
- **Allocation Normalization:** Predicted actions $a_1, \dots, a_5$ across the 5 states are normalized into percentage shares:
  $$\text{DQN Allocation } \% (s) = 100 \times \frac{a_s}{\sum_{k=1}^5 a_k}$$

---

## 2. Repository Structure

```text
covid-vaccine-allocation-dqn/
├── data/
│   └── projections.csv                          # Primary dataset (155 rows, 5 states, Dec 2020)
├── src/
│   ├── __init__.py                              # Module init
│   ├── dqn_agent.py                             # PyTorch DQN, ReplayBuffer, QNetwork
│   ├── environment.py                           # StatesEnv environment & feature extractor
│   ├── train.py                                 # Training pipeline with --resume auto and periodic checkpoints
│   └── evaluate.py                              # Allocation generator, Table 3 validation, and plot generation
├── .gitignore
├── README.md                                    # Project documentation
└── requirements.txt                             # PyTorch dependencies
```

---

## 3. Dataset Information

The primary dataset is `data/projections.csv`. It contains 155 rows representing 31 days (1–31 December 2020) across five Indian states:
- **Assam** (Moderate incidence)
- **Delhi** (High incidence)
- **Jharkhand** (Low incidence)
- **Maharashtra** (High incidence)
- **Nagaland** (Low incidence)

**Data Splits:**
- **Training Set:** 1 December to 26 December 2020 (130 state-day samples).
- **Evaluation / Test Set:** 27 December to 31 December 2020 (25 state-day samples).

---

## 4. Google Colab Execution Guide

All model checkpoints, metrics, and plots are saved directly to Google Drive:
`/content/drive/MyDrive/RL_Project`

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone Your Repository and Install Dependencies
```bash
# Clone your repository
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/VacSIM.git
%cd VacSIM

# Install minimal lightweight dependencies
!pip install -r requirements.txt
```

### Step 3: Start DQN Training
Train for 10,000 iterations (as specified in the paper Table 2). Checkpoints are saved every 1,000 iterations to Google Drive:
```bash
!python src/train.py \
    --data_path data/projections.csv \
    --output_dir /content/drive/MyDrive/RL_Project \
    --iterations 10000 \
    --save_interval 1000 \
    --buckets 1000
```

### Step 4: Resuming Interrupted Training
If your Colab GPU session disconnects or expires, simply re-run with `--resume auto`. It automatically loads `latest.pt` from Drive and resumes from the exact saved iteration:
```bash
!python src/train.py \
    --data_path data/projections.csv \
    --output_dir /content/drive/MyDrive/RL_Project \
    --iterations 10000 \
    --save_interval 1000 \
    --buckets 1000 \  
    --resume auto
```  

### Step 5: Run Evaluation and Generate Allocations
Evaluate the trained DQN model against the Naive baseline on the test period (27–31 December 2020):
```bash
!python src/evaluate.py \
    --data_path data/projections.csv \
    --output_dir /content/drive/MyDrive/RL_Project \
    --buckets 1000
```

---

## 5. Local Execution (Quick Test)

You can also run training and evaluation locally on CPU or GPU:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training
python src/train.py --iterations 1000 --save_interval 200 --output_dir ./outputs

# 3. Resume training
python src/train.py --iterations 2000 --save_interval 200 --output_dir ./outputs --resume auto

# 4. Run evaluation
python src/evaluate.py --output_dir ./outputs
```

---

## 6. Output Artifacts

All outputs in `/content/drive/MyDrive/RL_Project/` (or `./outputs/` locally) are structured as follows:

```text
RL_Project/
├── checkpoints/
│   ├── latest.pt              # Complete state (weights, target, optimizer, buffer, epsilon, step)
│   └── best.pt                # Checkpoint achieving highest rolling evaluation reward
└── results/
    ├── training_rewards.csv   # Reward trajectory over iterations
    ├── allocation_results.csv # Date-by-date allocation (% and vials) across all 5 states
    ├── evaluation_metrics.csv # DQN vs Naive baseline MAE and average reward metrics
    └── allocation_comparison.png # Comparison bar plot (DQN vs Naive vs Susceptible)
```
