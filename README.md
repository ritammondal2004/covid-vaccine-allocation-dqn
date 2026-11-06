# VaxOpt: DQN-Based COVID-19 Vaccine Allocation

**VaxOpt** is a reinforcement learning framework that formulates COVID-19 vaccine distribution across Indian states as a Markov Decision Process (MDP) and learns optimal allocation strategies using Deep Q-Networks (DQN).

---

## 1. Project Overview & Research Reference

This project is an academic reproduction and modern re-implementation of the DQN vaccine allocation methodology from:

> **Awasthi et al. (2022)**, *Learning Effective Strategies for COVID-19 Vaccine Distribution using Reinforcement Learning.*  
> *Intelligence-Based Medicine*, Volume 6, 100060.  
> [ScienceDirect Paper Link](https://doi.org/10.1016/j.ibmed.2022.100060)

### Data Source

- The projection dataset (`data/projections.csv`) is sourced from the authors' repository: [https://github.com/tavlab-iiitd/VacSIM](https://github.com/tavlab-iiitd/VacSIM).
- It provides Compartmental SEIR model projections combined with demographic data from the Census of India and healthcare capacity indicators from Development Data Lab (SHRUG).

---

## 2. Mathematical RL Formulation (Markov Decision Process)

The vaccine allocation task is modeled as an episodic Markov Decision Process (MDP) defined by the tuple:

⟨𝒮, 𝒜, 𝒫, ℛ, γ⟩

### A. State Space (𝒮 ⊂ ℝ<sup>9</sup>)

For each state i on date t, the agent observes a continuous 9-dimensional context vector s ∈ [0, 1]<sup>9</sup>:

| # | Context Feature | Symbol | Normalization Formula |
|:---:|---|:---:|---|
| 1 | Confirmed Cases Ratio | C<sub>i</sub> | Confirmed<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> Confirmed<sub>k</sub> |
| 2 | Predicted Death Rate | dr<sub>i</sub> | Deaths<sub>i</sub> / Confirmed<sub>i</sub> |
| 3 | Predicted Recovery Rate | rr<sub>i</sub> | Recoveries<sub>i</sub> / Confirmed<sub>i</sub> |
| 4 | Population Ratio | P<sub>i</sub> | Population<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> Population<sub>k</sub> |
| 5 | Predicted Susceptible Ratio | S<sub>i</sub> | Susceptible<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> Susceptible<sub>k</sub> |
| 6 | Hospital Beds Ratio | B<sub>i</sub> | Hospital Beds<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> Hospital Beds<sub>k</sub> |
| 7 | ICU Beds Ratio | ICU<sub>i</sub> | ICU Beds<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> ICU Beds<sub>k</sub> |
| 8 | Ventilators Ratio | V<sub>i</sub> | Ventilators<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> Ventilators<sub>k</sub> |
| 9 | Age Distribution (>50 yrs) Ratio | A<sub>50,i</sub> | Age > 50<sub>i</sub> / Σ<sub>k=1</sub><sup>5</sup> Age > 50<sub>k</sub> |

### B. Action Space (𝒜)

- The action space is discretized into n discrete allocation buckets:

  𝒜 = {0, 1, 2, ..., n − 1}, where n = 1000 (default)

- An action a<sub>i</sub> ∈ 𝒜 corresponds to allocating a fraction A<sub>i</sub> of the total available vaccine pool:

  A<sub>i</sub> = a<sub>i</sub> / n

- Doses allotted in one round of distribution:

  V<sub>i</sub> = A<sub>i</sub> × Batch Size, where Batch Size = 1,000,000 vials (10 lakh)

### C. Reward Function (ℛ)

The reward function optimizes vaccine distribution by penalizing any discrepancy between the allocated vaccine fraction (A<sub>i</sub>) and the state's share of the susceptible population (S<sub>i</sub>):

R<sub>i</sub>(s, a) = 100 × exp(−(A<sub>i</sub> − S<sub>i</sub>)² / 10<sup>−4</sup>) = 100 × exp(−10000 × (A<sub>i</sub> − S<sub>i</sub>)²)

- **Maximum Reward (R = 100):** Achieved when the allocation matches the susceptible share exactly (A<sub>i</sub> = S<sub>i</sub>).
- **Exponential Penalty:** Because the variance parameter is σ² = 10<sup>−4</sup> / 2, even a 1% deviation (|A<sub>i</sub> − S<sub>i</sub>| = 0.01) reduces the reward to 100 × e<sup>−1</sup> ≈ 36.79, and a 2% deviation reduces it to 1.83.

### D. Transition Dynamics & Episode Structure

- Administering V<sub>i</sub> doses transitions the state's susceptible count:

  Susceptible'<sub>i</sub> = max(0, Susceptible<sub>i</sub> − V<sub>i</sub>)

- Feature index 4 (susceptible ratio) updates to reflect the remaining proportion:

  S'<sub>i</sub> = (Susceptible<sub>i</sub> − V<sub>i</sub>) / (Σ<sub>k</sub> Susceptible<sub>k</sub> − V<sub>i</sub>)

- The episode terminates after 1 allocation step per state per date (`done = True`).

### E. Deep Q-Network (DQN) & Policy Update Formulas

The Q-network approximates the optimal action-value function Q<sup>*</sup>(s, a), satisfying the Bellman Optimality Equation:

Q<sup>*</sup>(s, a) = E[r + γ max<sub>a' ∈ 𝒜</sub> Q<sup>*</sup>(s', a') | s, a]

1. **Network Architecture:**

   9 → Linear → 64 → ReLU → 64 → Linear → 64 → ReLU → 64 → Linear → 1000

2. **Temporal Difference (TD) Target with Target Network θ<sup>−</sup>:**

   y = r + (1 − done) × γ × max<sub>a' ∈ 𝒜</sub> Q(s', a'; θ<sup>−</sup>)

3. **Loss Function:**

   L(θ) = (1 / B) Σ<sub>j=1</sub><sup>B</sup> (y<sub>j</sub> − Q(s<sub>j</sub>, a<sub>j</sub>; θ))²

4. **Target Network Sync:**

   Weights θ<sup>−</sup> are synchronized with the active policy network θ every 200 optimization steps.

5. **Exploration Strategy:**

   ε-greedy policy starting at ε<sub>0</sub> = 1.0 and linearly annealed down to ε<sub>min</sub> = 0.10 (10% exploration, matching paper Table 2).

### F. Multi-State Allocation Rule & Naive Baseline

1. **DQN Normalized Allocation:**

   On any test date, actions a<sub>1</sub>, ..., a<sub>5</sub> predicted greedily across the 5 states are normalized into percentage shares:

   DQN Allocation % (s) = 100 × a<sub>s</sub> / Σ<sub>k=1</sub><sup>5</sup> a<sub>k</sub>

2. **Naive Baseline Policy (Paper Section 3.4):**

   Allocates vaccines strictly proportional to confirmed infected cases:

   Naive Allocation % (s) = 100 × Confirmed<sub>s</sub> / Σ<sub>k=1</sub><sup>5</sup> Confirmed<sub>k</sub>

---

## 3. Dataset Description

The primary dataset is `data/projections.csv`. It contains 155 rows spanning 31 days (1–31 December 2020) across five representative Indian states:

- **Assam** (Moderate incidence)
- **Delhi** (High incidence)
- **Jharkhand** (Low incidence)
- **Maharashtra** (High incidence)
- **Nagaland** (Low incidence)

### Data Splits

- **Training Contexts (1 Dec – 26 Dec 2020):** 26 days × 5 states = **130 state-days**. The agent samples from these contexts during policy learning.
- **Testing Contexts (27 Dec – 31 Dec 2020):** 5 days × 5 states = **25 state-days**. These are unseen future dates used strictly for evaluation with frozen weights.

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

### Metric Summary

| Metric | DQN Policy | Naive Baseline Policy | Interpretation |
|:---|:---:|:---:|:---|
| **Average Reward** | **89.25** | 19.81 | **4.5× higher reward**; allocations closely track susceptible share. |
| **Mean Absolute Error to Susceptible (%)** | **0.31%** | 5.79% | DQN reduces distribution disparity by **over 18×**. |

### State-Wise Allocation on 31 December 2020

| State | Susceptible Share (%) | Naive Allocation (%) | DQN Allocation (%) | DQN Doses (Vials) |
|:---|:---:|:---:|:---:|:---:|
| **Assam** | 15.98% | 15.98% | **16.35%** | 163,471 |
| **Delhi** | 8.50% | 10.86% | **8.27%** | 82,745 |
| **Jharkhand** | **17.35%** | **5.31%** (Severe deficit) | **16.85%** | 168,517 |
| **Maharashtra** | 57.13% | 67.45% (Over-allocation) | **57.62%** | 576,186 |
| **Nagaland** | 1.04% | 0.40% | **0.91%** | 9,082 |

### Key Policy Takeaway

- **The Failure of the Naive Approach:** Allocating doses strictly by current confirmed cases severely disadvantages states like **Jharkhand** (allocated only 5.31% despite having 17.35% of the susceptible population) and excessively concentrates doses in **Maharashtra** (67.45%).
- **The DQN Advantage:** By incorporating susceptible populations and healthcare capacities into the state space, the DQN policy learns an equitable distribution with an error of just **0.31%**, preventing supply shortages in high-vulnerability regions.
