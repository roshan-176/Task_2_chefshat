import sys
import os
import asyncio

CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)                 
PROJECT_DIR = os.path.dirname(SRC_DIR)                  
CHEFSHAT_ROOT = os.path.dirname(PROJECT_DIR)            

CHEFSHAT_GYM_SRC = os.path.join(
    CHEFSHAT_ROOT,
    "ChefsHatGYM-main",
    "src"
)

sys.path.insert(0, CHEFSHAT_GYM_SRC)


from src.chefshat_env.rooms.room import Room
from agents.random_agent import RandomAgent


async def main():
    room = Room(
        run_remote_room=False,
        room_name="test_room",
        max_matches=1
    )

    players = [
        RandomAgent(name=f"P{i}", log_directory=room.room_dir)
        for i in range(4)
    ]

    for p in players:
        room.connect_player(p)

    await room.run()
    print("Final scores:", room.final_scores)


asyncio.run(main())