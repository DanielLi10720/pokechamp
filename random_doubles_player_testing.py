import sys
import asyncio
sys.path.append("../src")

from poke_env.player.player import Player
from poke_env.player import RandomDoublesPlayer
from poke_env.data import GenData

# The RandomPlayer is a basic agent that makes decisions randomly,
# serving as a starting point for more complex agent development.
random_player = RandomDoublesPlayer()
second_player = RandomDoublesPlayer()

# The battle_against method initiates a battle between two players.
# Here we are using asynchronous programming (await) to start the battle.
async def main():
    await random_player.battle_against(second_player, n_battles=1)

asyncio.run(main())