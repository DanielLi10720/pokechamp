"""This module defines a random doubles player baseline
"""

from poke_env.environment import DoubleBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player


class RandomDoublesPlayer(Player):
    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        return self.choose_random_doubles_move(battle)