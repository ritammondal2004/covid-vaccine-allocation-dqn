"""
Training script for DQN-based COVID-19 Vaccine Allocation (VacSIM).
Supports automatic checkpoint saving, periodic Drive syncing, and seamless resume.
"""

import argparse
import os
import sys
import time
from collections import deque
import numpy as np
import pandas as pd
from tqdm import tqdm

from environment import extract_context_data, create_env_from_record
from dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN for COVID-19 Vaccine Allocation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/projections.csv",
        help="Path to projections.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save checkpoints and logs (e.g., /content/drive/MyDrive/RL_Project)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Total training iterations (paper specifies 10,000 for DQN)",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=1000,
        help="Discretization buckets for action space (default 1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for DQN gradient update",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Interval of iterations at which to save checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help="'auto' to resume from latest.pt, or path to specific checkpoint, or 'none'",
    )
    parser.add_argument(
        "--train_start_date",
        type=str,
        default="2020-12-01",
        help="Start date for training window (inclusive)",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        default="2020-12-26",
        help="End date for training window (inclusive, 26 days)",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    # Setup directories
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    print("=" * 70)
    print("VacSIM DQN Training — COVID-19 Vaccine Allocation")
    print(f"Data file:         {args.data_path}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Action buckets:    {args.buckets}")
    print(f"Target iterations: {args.iterations}")
    print(f"Save interval:     {args.save_interval}")
    print("=" * 70)

    # Load context data from CSV
    if not os.path.exists(args.data_path):
        # Check parent or relative directory
        alt_path = os.path.join("..", args.data_path)
        if os.path.exists(alt_path):
            args.data_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find dataset at {args.data_path}")

    dates, state_names, daily_contexts = extract_context_data(args.data_path)

    # Filter training dates (1 Dec to 26 Dec as per paper Section 3.1 & 3.2.2)
    train_dates = [d for d in dates if args.train_start_date <= d <= args.train_end_date]
    print(f"Training across {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]}).")
    print(f"States: {', '.join(state_names)}")

    # Flatten all state-day context records into a training pool
    train_records = []
    for d in train_dates:
        train_records.extend(daily_contexts[d])
    num_contexts = len(train_records)
    print(f"Total training context records: {num_contexts}")

    # Initialize DQN Agent
    agent = DQNAgent(
        state_dim=9,
        action_dim=args.buckets,
        learning_rate=args.lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay_steps=max(2000, int(args.iterations * 0.4)),
        buffer_capacity=20000,
        batch_size=args.batch_size,
        target_update_interval=200,
    )
    print(f"Agent initialized on compute device: {agent.device}")

    start_iteration = 0
    best_avg_reward = -float("inf")
    reward_history = []
    loss_history = []

    # Handle resume logic
    resume_path = None
    if args.resume.lower() == "auto":
        if os.path.exists(latest_ckpt_path):
            resume_path = latest_ckpt_path
        else:
            print("[Resume Auto] No existing checkpoint found at latest.pt. Starting fresh training.")
    elif args.resume.lower() != "none" and args.resume != "":
        if os.path.exists(args.resume):
            resume_path = args.resume
        elif os.path.exists(os.path.join(checkpoint_dir, args.resume)):
            resume_path = os.path.join(checkpoint_dir, args.resume)
        else:
            print(f"[Warning] Specified resume file not found: {args.resume}. Starting fresh.")

    if resume_path:
        print(f"[Resume] Loading checkpoint from: {resume_path}")
        extra_info = agent.load_checkpoint(resume_path)
        start_iteration = extra_info.get("iteration", 0)
        best_avg_reward = extra_info.get("best_avg_reward", -float("inf"))
        reward_history = extra_info.get("reward_history", [])
        loss_history = extra_info.get("loss_history", [])
        print(f"[Resume] Successfully resumed from iteration {start_iteration} with epsilon={agent.epsilon:.3f}")

    if start_iteration >= args.iterations:
        print(f"Training already completed up to {start_iteration} iterations (target {args.iterations}).")
        return

    # Warm-up replay buffer if empty
    if len(agent.replay_buffer) < args.batch_size:
        print("[Warm-up] Seeding experience replay buffer with exploratory transitions...")
        for _ in range(max(200, args.batch_size * 2)):
            rec = train_records[np.random.randint(num_contexts)]
            env = create_env_from_record(rec, buckets=args.buckets)
            s = env.reset()
            a = np.random.randint(args.buckets)
            ns, r, done, _ = env.step(a)
            agent.replay_buffer.push(s, a, r, ns, done)

    print(f"Starting training loop from iteration {start_iteration + 1} to {args.iterations}...")
    pbar = tqdm(range(start_iteration + 1, args.iterations + 1), initial=start_iteration, total=args.iterations)

    rolling_reward = deque(maxlen=100)
    rolling_loss = deque(maxlen=100)

    for iteration in pbar:
        # Sample a random state-day record to maintain spatial independence across states
        rec = train_records[np.random.randint(num_contexts)]
        env = create_env_from_record(rec, buckets=args.buckets)
        state = env.reset()

        action = agent.select_action(state, evaluate=False)
        next_state, reward, done, _ = env.step(action)

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.update()

        rolling_reward.append(reward)
        if loss is not None:
            rolling_loss.append(loss)

        # Update progress bar
        avg_r = np.mean(rolling_reward) if rolling_reward else 0.0
        avg_l = np.mean(rolling_loss) if rolling_loss else 0.0
        pbar.set_postfix({"Reward": f"{avg_r:.1f}", "Loss": f"{avg_l:.4f}", "Eps": f"{agent.epsilon:.2f}"})

        # Periodic checkpoint save
        if iteration % args.save_interval == 0 or iteration == args.iterations:
            reward_history.append((iteration, float(avg_r)))
            if rolling_loss:
                loss_history.append((iteration, float(avg_l)))

            extra_info = {
                "iteration": iteration,
                "best_avg_reward": best_avg_reward,
                "reward_history": reward_history,
                "loss_history": loss_history,
                "args": vars(args),
            }
            agent.save_checkpoint(latest_ckpt_path, extra_info=extra_info)
            tqdm.write(f"[{iteration}/{args.iterations}] Saved checkpoint to: {latest_ckpt_path} (Avg Reward: {avg_r:.2f})")

            # Check for best model
            if avg_r > best_avg_reward and iteration >= int(args.iterations * 0.2):
                best_avg_reward = avg_r
                extra_info["best_avg_reward"] = best_avg_reward
                agent.save_checkpoint(best_ckpt_path, extra_info=extra_info)
                tqdm.write(f"[{iteration}] New best checkpoint saved with Avg Reward: {best_avg_reward:.2f}")

    # Final summary and save logs
    print("\nTraining completed successfully!")
    final_info = {
        "iteration": args.iterations,
        "best_avg_reward": best_avg_reward,
        "reward_history": reward_history,
        "loss_history": loss_history,
    }
    agent.save_checkpoint(latest_ckpt_path, extra_info=final_info)

    log_df = pd.DataFrame(reward_history, columns=["Iteration", "AverageReward"])
    log_path = os.path.join(results_dir, "training_rewards.csv")
    log_df.to_csv(log_path, index=False)
    print(f"Training reward history logged to: {log_path}")
    print(f"Final model checkpoint: {latest_ckpt_path}")


if __name__ == "__main__":
    train()
