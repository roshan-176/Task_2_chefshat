import asyncio
import random
import numpy as np
import torch

from src.chefshat_env.rooms.room import Room
from src.chefshat_env.agents.random_agent import RandomAgent
from src.rl_agents.chefshat_dqn_wrapper import ChefsHatDQNAgent


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


async def train():
    room = Room(
        run_remote_room=False,
        room_name="training_room",
        max_matches=200
    )

    dqn_agent = ChefsHatDQNAgent(
        name="DQN_Player",
        log_directory=room.room_dir,
        training=True
    )

    opponents = [
        RandomAgent(name=f"R{i}", log_directory=room.room_dir)
        for i in range(3)
    ]

    room.connect_player(dqn_agent)
    for opp in opponents:
        room.connect_player(opp)

    print("Starting training...")
    await room.run()
    print("Training finished:", room.final_scores)

    torch.save(
        dqn_agent.agent.q_net.state_dict(),
        "results/dqn_model.pt"
    )


if __name__ == "__main__":
    asyncio.run(train())