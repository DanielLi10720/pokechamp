import sys

sys.path.append("../src")

from poke_env import RandomPlayer
from poke_env.data import GenData

# The RandomPlayer is a basic agent that makes decisions randomly,
# serving as a starting point for more complex agent development.
random_player = RandomPlayer()
second_player = RandomPlayer()

# The battle_against method initiates a battle between two players.
# Here we are using asynchronous programming (await) to start the battle.
await random_player.battle_against(second_player, n_battles=1)