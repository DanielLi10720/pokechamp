import sys
import asyncio
from poke_env import RandomPlayer

sys.path.append("../src")

# Define the battle logic inside an async function
async def main():
    random_player = RandomPlayer()
    second_player = RandomPlayer()

    await random_player.battle_against(second_player, n_battles=1)

# Run the async function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
