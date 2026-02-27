import asyncio
import random
import numpy as np

from src.chefshat_env.rooms.room import Room
from src.rl_agents.chefshat_dqn_wrapper import ChefsHatDQNAgent
from src.chefshat_env.agents.random_agent import RandomAgent


async def evaluate(seed: int):
    print(f"\n========== Evaluating with seed {seed} ==========")

    # --- Set seeds for robustness ---
    random.seed(seed)
    np.random.seed(seed)

    # --- Create room ---
    room = Room(
        run_remote_room=False,
        room_name=f"eval_seed_{seed}",
        max_matches=50
    )

    # --- Create agents ---
    dqn_agent = ChefsHatDQNAgent(
        name="DQN_Player",
        log_directory=room.room_dir,
        training=False
    )

    opponents = [
        RandomAgent(name="R0", log_directory=room.room_dir),
        RandomAgent(name="R1", log_directory=room.room_dir),
        RandomAgent(name="R2", log_directory=room.room_dir),
    ]

    # --- Connect agents ---
    room.connect_player(dqn_agent)
    for opp in opponents:
        room.connect_player(opp)

    # --- Run evaluation ---
    await room.run()

    print(f"Final scores (seed {seed}): {room.final_scores}")
    return room.final_scores


async def main():
    seeds = [0, 1, 2, 3, 4]
    results = []

    for seed in seeds:
        scores = await evaluate(seed)
        results.append(scores)

    print("\n========== ROBUSTNESS SUMMARY ==========")
    for seed, scores in zip(seeds, results):
        print(f"Seed {seed} → {scores}")


if __name__ == "__main__":
    asyncio.run(main())