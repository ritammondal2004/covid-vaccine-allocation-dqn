"""
Evaluation and Allocation Generation script for VaxOpt.
Loads trained DQN checkpoints, computes normalized vaccine distributions across states,
compares against Naive (incidence-based) baseline and published Table 3 results,
and outputs evaluation CSVs and comparison plots.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import extract_context_data, create_env_from_record
from dqn_agent import DQNAgent


# Reference values published in Table 3 of Awasthi et al. (2022) for DQN + CB
TABLE_3_PUBLISHED = {
    200: {"Assam": 15.8974, "Delhi": 8.2051, "Jharkhand": 16.9231, "Maharashtra": 58.4615, "Nagaland": 0.5128},
    300: {"Assam": 15.9864, "Delhi": 8.1633, "Jharkhand": 17.0068, "Maharashtra": 58.1633, "Nagaland": 0.6803},
    400: {"Assam": 16.1616, "Delhi": 8.3333, "Jharkhand": 17.1717, "Maharashtra": 57.5758, "Nagaland": 0.7576},
    500: {"Assam": 16.1290, "Delhi": 8.2661, "Jharkhand": 17.1371, "Maharashtra": 57.6613, "Nagaland": 0.8065},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQN Vaccine Allocation Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/projections.csv",
        help="Path to projections.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (if omitted, searches output_dir/checkpoints/best.pt then latest.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save evaluation results and plots",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=1000,
        help="Discretization bucket size",
    )
    parser.add_argument(
        "--test_start_date",
        type=str,
        default="2020-12-27",
        help="Start date for test evaluation (inclusive)",
    )
    parser.add_argument(
        "--test_end_date",
        type=str,
        default="2020-12-31",
        help="End date for test evaluation (inclusive, 5 days)",
    )
    parser.add_argument(
        "--total_vials",
        type=float,
        default=1000000.0,
        help="Total vaccine batch vials (10 lakh doses)",
    )
    return parser.parse_args()


def locate_checkpoint(args):
    """Finds best.pt or latest.pt if no explicit checkpoint is passed."""
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        return args.checkpoint_path

    search_paths = [
        os.path.join(args.output_dir, "checkpoints", "best.pt"),
        os.path.join(args.output_dir, "checkpoints", "latest.pt"),
        os.path.join("outputs", "checkpoints", "best.pt"),
        os.path.join("outputs", "checkpoints", "latest.pt"),
        os.path.join("checkpoints", "best.pt"),
        os.path.join("checkpoints", "latest.pt"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None


def evaluate():
    args = parse_args()
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = locate_checkpoint(args)
    if not ckpt_path:
        print("[Error] No model checkpoint found. Train the model first using train.py.")
        sys.exit(1)

    print("=" * 70)
    print("VaxOpt DQN Model Evaluation")
    print(f"Checkpoint:       {ckpt_path}")
    print(f"Dataset:          {args.data_path}")
    print(f"Evaluation Dates: {args.test_start_date} to {args.test_end_date}")
    print(f"Results Output:   {results_dir}")
    print("=" * 70)

    # Load dataset
    dates, state_names, daily_contexts = extract_context_data(args.data_path)
    test_dates = [d for d in dates if args.test_start_date <= d <= args.test_end_date]
    if not test_dates:
        # Fallback to last 5 available dates
        test_dates = dates[-5:]
        print(f"[Note] Target dates not found. Using last 5 dates: {test_dates[0]} to {test_dates[-1]}")

    # Load DQN Agent
    agent = DQNAgent(state_dim=9, action_dim=args.buckets)
    agent.load_checkpoint(ckpt_path)
    print(f"Loaded trained DQN Agent successfully from: {ckpt_path}")

    records_out = []
    daily_dqn_rewards = []
    daily_naive_rewards = []
    daily_dqn_mae = []
    daily_naive_mae = []

    for date in test_dates:
        day_records = daily_contexts[date]
        total_confirmed = sum(r["raw_confirmed"] for r in day_records)
        total_susceptible = sum(r["raw_susceptible"] for r in day_records)

        unadjusted_actions = []
        raw_envs = []
        for r in day_records:
            env = create_env_from_record(r, buckets=args.buckets, total_vials=args.total_vials)
            state = env.reset()
            action = agent.select_action(state, evaluate=True)
            unadjusted_actions.append(action)
            raw_envs.append(env)

        sum_actions = max(sum(unadjusted_actions), 1)

        for s_idx, r in enumerate(day_records):
            state_name = r["state"]
            action = unadjusted_actions[s_idx]

            # DQN allocation percentage: action_s / sum(actions) * 100
            dqn_pct = 100.0 * float(action) / float(sum_actions)
            dqn_doses = (dqn_pct / 100.0) * args.total_vials

            # Naive baseline: % of vaccine given = % of infected cases in state (Section 3.4)
            naive_pct = 100.0 * float(r["raw_confirmed"]) / float(total_confirmed)
            naive_doses = (naive_pct / 100.0) * args.total_vials

            # Ideal allocation proportional to susceptible population
            susc_pct = 100.0 * float(r["raw_susceptible"]) / float(total_susceptible)

            # Paper reward function: R = 100 * exp(-(A - S)^2 / 1e-4)
            dqn_a = dqn_pct / 100.0
            naive_a = naive_pct / 100.0
            s_frac = susc_pct / 100.0

            r_dqn = 100.0 * np.exp(-((dqn_a - s_frac) ** 2) / 0.0001)
            r_naive = 100.0 * np.exp(-((naive_a - s_frac) ** 2) / 0.0001)

            daily_dqn_rewards.append(r_dqn)
            daily_naive_rewards.append(r_naive)
            daily_dqn_mae.append(abs(dqn_pct - susc_pct))
            daily_naive_mae.append(abs(naive_pct - susc_pct))

            records_out.append(
                {
                    "Date": date,
                    "State": state_name,
                    "DQN_Action_Bucket": action,
                    "DQN_Allocation_Pct": round(dqn_pct, 4),
                    "DQN_Doses": round(dqn_doses, 1),
                    "Naive_Allocation_Pct": round(naive_pct, 4),
                    "Naive_Doses": round(naive_doses, 1),
                    "Susceptible_Pct": round(susc_pct, 4),
                    "DQN_Reward": round(float(r_dqn), 4),
                    "Naive_Reward": round(float(r_naive), 4),
                }
            )

    results_df = pd.DataFrame(records_out)
    res_path = os.path.join(results_dir, "allocation_results.csv")
    results_df.to_csv(res_path, index=False)
    print(f"\nSaved daily allocations to: {res_path}")

    # Aggregated Summary by State on the last test date (31 Dec 2020)
    last_date = test_dates[-1]
    last_df = results_df[results_df["Date"] == last_date]

    print("\n" + "=" * 70)
    print(f"Vaccine Allocation on {last_date} (DQN vs Naive vs Susceptible)")
    print("=" * 70)
    summary_cols = [
        "State",
        "DQN_Allocation_Pct",
        "Naive_Allocation_Pct",
        "Susceptible_Pct",
        "DQN_Doses",
    ]
    print(last_df[summary_cols].to_string(index=False))

    # Compare with Table 3 if bucket matches
    if args.buckets in TABLE_3_PUBLISHED:
        print("\n" + "-" * 70)
        print(f"Comparison with Paper Table 3 (DQN + CB, Bucket {args.buckets}):")
        ref = TABLE_3_PUBLISHED[args.buckets]
        for st in state_names:
            model_val = last_df.loc[last_df["State"] == st, "DQN_Allocation_Pct"].values[0]
            pub_val = ref.get(st, float("nan"))
            print(f"  {st:12s}: Model = {model_val:7.4f}% | Published = {pub_val:7.4f}% (Diff: {abs(model_val-pub_val):.4f}%)")
        print("-" * 70)

    # Metrics Summary
    avg_dqn_r = float(np.mean(daily_dqn_rewards))
    avg_naive_r = float(np.mean(daily_naive_rewards))
    avg_dqn_err = float(np.mean(daily_dqn_mae))
    avg_naive_err = float(np.mean(daily_naive_mae))

    metrics_df = pd.DataFrame(
        [
            {"Metric": "Average DQN Reward", "Value": round(avg_dqn_r, 4)},
            {"Metric": "Average Naive Reward", "Value": round(avg_naive_r, 4)},
            {"Metric": "DQN Mean Abs Error to Susceptible (%)", "Value": round(avg_dqn_err, 4)},
            {"Metric": "Naive Mean Abs Error to Susceptible (%)", "Value": round(avg_naive_err, 4)},
        ]
    )
    metrics_path = os.path.join(results_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nEvaluation metrics saved to: {metrics_path}")
    print(metrics_df.to_string(index=False))

    # Generate Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(state_names))
    width = 0.28

    dqn_vals = [last_df.loc[last_df["State"] == s, "DQN_Allocation_Pct"].values[0] for s in state_names]
    naive_vals = [last_df.loc[last_df["State"] == s, "Naive_Allocation_Pct"].values[0] for s in state_names]
    susc_vals = [last_df.loc[last_df["State"] == s, "Susceptible_Pct"].values[0] for s in state_names]

    ax.bar(x - width, naive_vals, width, label="Naive Policy (% of Infected)", color="#ff7f0e")
    ax.bar(x, dqn_vals, width, label="DQN Policy (Learned)", color="#1f77b4")
    ax.bar(x + width, susc_vals, width, label="Susceptible Population Share (%)", color="#2ca02c")

    ax.set_ylabel("Percentage Distribution (%)", fontsize=12)
    ax.set_title(f"COVID-19 Vaccine Allocation on {last_date} (DQN vs Naive Baseline)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(state_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plot_path = os.path.join(results_dir, "allocation_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Allocation comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    evaluate()

