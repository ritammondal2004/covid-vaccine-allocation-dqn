# VaxOpt: DQN-Based COVID-19 Vaccine Allocation

**VaxOpt** is a reinforcement learning framework that formulates COVID-19 vaccine distribution across Indian states as a Markov Decision Process (MDP) and learns optimal allocation strategies using Deep Q-Networks (DQN).

---

## 1. Project Overview & Research Reference

This project is an academic reproduction and modern re-implementation of the DQN vaccine allocation methodology from:

> **Awasthi et al. (2022)**, *Learning Effective Strategies for COVID-19 Vaccine Distribution using Reinforcement Learning.*  
> *Intelligence-Based Medicine*, Volume 6, 100060.  
> [ScienceDirect Paper Link](https://doi.org/10.1016/j.ibmed.2022.100060)

### Data Source
* The projection dataset (`data/projections.csv`) is sourced from the authors' repository: [https://github.com/tavlab-iiitd/VacSIM](https://github.com/tavlab-iiitd/VacSIM).
* It provides Compartmental SEIR model projections combined with demographic data from the Census of India and healthcare capacity indicators from Development Data Lab (SHRUG).

---

## 2. Mathematical RL Formulation (Markov Decision Process)

The vaccine allocation task is modeled as an episodic Markov Decision Process (MDP) defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

### A. State Space ($\mathcal{S} \subset \mathbb{R}^9$)
For each state $i$ on date $t$, the agent observes a continuous 9-dimensional context vector $s \in [0, 1]^9$:

| # | Context Feature | Symbol | Normalization Formula |
|:---:|---|:---:|:---:|
| 1 | Confirmed Cases Ratio | $C_i$ | $\frac{\text{Confirmed}_i}{\sum_{k=1}^5 \text{Confirmed}_k}$ |
| 2 | Predicted Death Rate | $dr_i$ | $\frac{\text{Deaths}_i}{\text{Confirmed}_i}$ |
| 3 | Predicted Recovery Rate | $rr_i$ | $\frac{\text{Recoveries}_i}{\text{Confirmed}_i}$ |
| 4 | Population Ratio | $P_i$ | $\frac{\text{Population}_i}{\sum_{k=1}^5 \text{Population}_k}$ |
| 5 | Predicted Susceptible Ratio | $S_i$ | $\frac{\text{Susceptible}_i}{\sum_{k=1}^5 \text{Susceptible}_k}$ |
| 6 | Hospital Beds Ratio | $B_i$ | $\frac{\text{Hospital Beds}_i}{\sum_{k=1}^5 \text{Hospital Beds}_k}$ |
| 7 | ICU Beds Ratio | $ICU_i$ | $\frac{\text{ICU Beds}_i}{\sum_{k=1}^5 \text{ICU Beds}_k}$ |
| 8 | Ventilators Ratio | $V_i$ | $\frac{\text{Ventilators}_i}{\sum_{k=1}^5 \text{Ventilators}_k}$ |
| 9 | Age Distribution (>50 yrs) Ratio | $A_{50, i}$ | $\frac{\text{Age}>50_i}{\sum_{k=1}^5 \text{Age}>50_k}$ |

### B. Action Space ($\mathcal{A}$)
* The action space is discretized into $n$ discrete allocation buckets:
  $$
  \mathcal{A} = \{0, 1, 2, \dots, n-1\}, \quad \text{where } n = 1000 \text{ (default)}
  $$
* An action $a_i \in \mathcal{A}$ corresponds to allocating a fraction $A_i$ of the total available vaccine pool:
  $$
  A_i = \frac{a_i}{n}
  $$
* Doses allotted in one round of distribution:
  $$
  V_i = A_i \times \text{Batch Size}, \quad \text{where Batch Size} = 1,000,000 \text{ vials (10 lakh)}
  $$

### C. Reward Function ($\mathcal{R}$)
The reward function optimizes vaccine distribution by penalizing any discrepancy between the allocated vaccine fraction ($A_i$) and the state's share of the susceptible population ($S_i$):

$$
R_i(s, a) = 100 \cdot \exp\left( - \frac{(A_i - S_i)^2}{10^{-4}} \right) = 100 \cdot \exp\left( - 10000 \cdot (A_i - S_i)^2 \right)
$$

* **Maximum Reward ($R = 100$):** Achieved when the allocation matches the susceptible share exactly ($A_i = S_i$).
* **Exponential Penalty:** Because the variance parameter is $\sigma^2 = \frac{10^{-4}}{2}$, even a 1% deviation ($|A_i - S_i| = 0.01$) reduces the reward to $100 \cdot e^{-1} \approx 36.79$, and a 2% deviation reduces it to $1.83$.

### D. Transition Dynamics & Episode Structure
* Administering $V_i$ doses transitions the state's susceptible count:
  $$
  \text{Susceptible}_i' = \max(0, \, \text{Susceptible}_i - V_i)
  $$
* Feature index 4 (susceptible ratio) updates to reflect the remaining proportion:
  $$
  S_i' = \frac{\text{Susceptible}_i - V_i}{\sum_k \text{Susceptible}_k - V_i}
  $$
* The episode terminates after 1 allocation step per state per date (`done = True`).

### E. Deep Q-Network (DQN) & Policy Update Formulas
The Q-network approximates the optimal action-value function $Q^*(s, a)$, satisfying the Bellman Optimality Equation:

$$
Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \;\middle|\; s, a \right]
$$

1. **Network Architecture:**
   $$
   9 \xrightarrow{\text{Linear}} 64 \xrightarrow{\text{ReLU}} 64 \xrightarrow{\text{Linear}} 64 \xrightarrow{\text{ReLU}} 64 \xrightarrow{\text{Linear}} 1000
   $$
2. **Temporal Difference (TD) Target with Target Network $\theta^-$:**
   $$
   y = r + (1 - \text{done}) \cdot \gamma \max_{a' \in \mathcal{A}} Q(s', a'; \theta^-)
   $$
3. **Loss Function:**
   $$
   L(\theta) = \frac{1}{B} \sum_{j=1}^B \left( y_j - Q(s_j, a_j; \theta) \right)^2
   $$
4. **Target Network Sync:**
   Weights $\theta^-$ are synchronized with the active policy network $\theta$ every 200 optimization steps.
5. **Exploration Strategy:**
   $\epsilon$-greedy policy starting at $\epsilon_0 = 1.0$ and linearly annealed down to $\epsilon_{\min} = 0.10$ (10% exploration, matching paper Table 2).

### F. Multi-State Allocation Rule & Naive Baseline
1. **DQN Normalized Allocation:**
   On any test date, actions $a_1, \dots, a_5$ predicted greedily across the 5 states are normalized into percentage shares:
   $$
   \text{DQN Allocation \%}(s) = 100 \times \frac{a_s}{\sum_{k=1}^5 a_k}
   $$
2. **Naive Baseline Policy (Paper Section 3.4):**
   Allocates vaccines strictly proportional to confirmed infected cases:
   $$
   \text{Naive Allocation \%}(s) = 100 \times \frac{\text{Confirmed}_s}{\sum_{k=1}^5 \text{Confirmed}_k}
   $$

---

## 3. Dataset Description

The primary dataset is `data/projections.csv`. It contains 155 rows spanning 31 days (1–31 December 2020) across five representative Indian states:
- **Assam** (Moderate incidence)
- **Delhi** (High incidence)
- **Jharkhand** (Low incidence)
- **Maharashtra** (High incidence)
- **Nagaland** (Low incidence)

### Data Splits:
* **Training Contexts (1 Dec – 26 Dec 2020):** 26 days $\times$ 5 states = **130 state-days**. The agent samples from these contexts during policy learning.
* **Testing Contexts (27 Dec – 31 Dec 2020):** 5 days $\times$ 5 states = **25 state-days**. These are unseen future dates used strictly for evaluation with frozen weights.

---

## 4. Repository Structure

```text
covid-vaccine-allocation-dqn/
├── data/
│   └── projections.csv          # Sole required dataset (155 rows, 5 states, Dec 2020)
├── src/
│   ├── __init__.py              # Package marker
│   ├── dqn_agent.py             # PyTorch DQN Agent, ReplayBuffer, QNetwork
│   ├── environment.py           # StatesEnv MDP environment & data extractor
│   ├── train.py                 # Training script with Drive checkpointing & auto-resume
│   └── evaluate.py              # Evaluator computing DQN vs Naive allocations & plots
├── .gitignore
├── LICENSE
├── README.md                    # Project documentation
└── requirements.txt             # Lightweight dependencies (torch, pandas, numpy, matplotlib, tqdm)
```

---

## 5. Step-by-Step Reproduction Guide

### Option A: Google Colab (Recommended)

1. **Mount Google Drive & Set Runtime:**
   In Google Colab, select `Runtime` → `Change runtime type` → **T4 GPU**, then run:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Clone Repository & Install Dependencies:**
   ```bash
   !git clone https://github.com/ritammondal2004/covid-vaccine-allocation-dqn.git
   %cd covid-vaccine-allocation-dqn
   !pip install -r requirements.txt
   ```

3. **Start DQN Training (10,000 Iterations):**
   Checkpoints and reward logs are saved directly to `/content/drive/MyDrive/RL_Project`:
   ```bash
   !python src/train.py \
       --data_path data/projections.csv \
       --output_dir /content/drive/MyDrive/RL_Project \
       --iterations 10000 \
       --save_interval 1000 \
       --buckets 1000
   ```

4. **Automatic Resume from Checkpoint:**
   If a Colab session disconnects or expires, simply re-run with `--resume auto`:
   ```bash
   !python src/train.py \
       --data_path data/projections.csv \
       --output_dir /content/drive/MyDrive/RL_Project \
       --iterations 10000 \
       --save_interval 1000 \
       --buckets 1000 \
       --resume auto
   ```
   *The script detects `latest.pt` on Google Drive, restores model weights, target network, optimizer, replay buffer, and iteration count, and continues training without losing progress.*

5. **Run Evaluation & Generate Allocations:**
   ```bash
   !python src/evaluate.py \
       --data_path data/projections.csv \
       --output_dir /content/drive/MyDrive/RL_Project \
       --buckets 1000
   ```

6. **Display Results in Colab:**
   ```python
   from IPython.display import Image, display
   import pandas as pd

   # 1. Display comparison bar plot
   display(Image('/content/drive/MyDrive/RL_Project/results/allocation_comparison.png'))

   # 2. View metrics summary
   print(pd.read_csv('/content/drive/MyDrive/RL_Project/results/evaluation_metrics.csv').to_string(index=False))
   ```

---

### Option B: Local Execution

```bash
# 1. Clone & setup
git clone https://github.com/ritammondal2004/covid-vaccine-allocation-dqn.git
cd covid-vaccine-allocation-dqn
pip install -r requirements.txt

# 2. Train model
python src/train.py --output_dir ./outputs --iterations 10000 --save_interval 1000

# 3. Resume training if needed
python src/train.py --output_dir ./outputs --iterations 10000 --resume auto

# 4. Evaluate allocations
python src/evaluate.py --output_dir ./outputs
```

---

## 6. Empirical Results & Findings

Evaluation on the test period (27–31 December 2020) demonstrates that the DQN policy significantly outperforms the incidence-based Naive policy:

### Metric Summary:
| Metric | DQN Policy | Naive Baseline Policy | Interpretation |
| :--- | :---: | :---: | :--- |
| **Average Reward** | **89.25** | 19.81 | **4.5× higher reward**; allocations closely track susceptible share. |
| **Mean Absolute Error to Susceptible (%)** | **0.31%** | 5.79% | DQN reduces distribution disparity by **over 18×**. |

### State-Wise Allocation on 31 December 2020:
| State | Susceptible Share (%) | Naive Allocation (%) | DQN Allocation (%) | DQN Doses (Vials) |
| :--- | :---: | :---: | :---: | :---: |
| **Assam** | 15.98% | 15.98% | **16.35%** | 163,471 |
| **Delhi** | 8.50% | 10.86% | **8.27%** | 82,745 |
| **Jharkhand** | **17.35%** | **5.31%** (Severe deficit) | **16.85%** | 168,517 |
| **Maharashtra** | 57.13% | 67.45% (Over-allocation) | **57.62%** | 576,186 |
| **Nagaland** | 1.04% | 0.40% | **0.91%** | 9,082 |

### Key Policy Takeaway:
* **The Failure of the Naive Approach:** Allocating doses strictly by current confirmed cases severely disadvantages states like **Jharkhand** (allocated only 5.31% despite having 17.35% of the susceptible population) and excessively concentrates doses in **Maharashtra** (67.45%).
* **The DQN Advantage:** By incorporating susceptible populations and healthcare capacities into the state space, the DQN policy learns an equitable distribution with an error of just **0.31%**, preventing supply shortages in high-vulnerability regions.
