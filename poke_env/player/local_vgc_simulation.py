import json
import sys
from time import sleep
from typing import Callable, Dict, List, Optional
import numpy as np
from copy import deepcopy

import orjson

from poke_env.data.gen_data import GenData
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer

# Avoid circular import by importing here
try:
    from pokechamp.data_cache import get_cached_moves_set
    from pokechamp.sim_constants import get_simulation_optimizer, TYPE_LIST
except ImportError:
    # Fallback if optimization modules are not available
    get_cached_moves_set = None
    get_simulation_optimizer = None
    TYPE_LIST = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

DEBUG = False

def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    TYPE_list = TYPE_LIST  # Use cached constant instead of recreating

    move_type_damage_multiplier_list = []

    if type_2:
        for type in TYPE_list:
            if 'STELLAR' not in [type, type_1, type_2]:
                move_type_damage_multiplier_list.append(type_chart[type_1][type] * type_chart[type_2][type])
            else: 
                move_type_damage_multiplier_list.append(1)
        move_type_damage_multiplier_dict = dict(zip(TYPE_list, move_type_damage_multiplier_list))
    else:
        if type_1 == 'STELLAR':
            move_type_damage_multiplier_dict = {'BUG': 1, 'DARK': 1, 'DRAGON': 1, 'ELECTRIC': 1, 'FAIRY': 1, 'FIGHTING': 1, 'FIRE': 1, 'FLYING': 1, 'GHOST': 1, 'GRASS': 1, 'GROUND': 1, 'ICE': 1, 'NORMAL': 1, 'POISON': 1, 'PSYCHIC': 1, 'ROCK': 1, 'STEEL': 1, 'WATER': 1}
        else:
            move_type_damage_multiplier_dict = type_chart[type_1]

    effective_type_list = []
    extreme_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    for type, value in move_type_damage_multiplier_dict.items():
        if value == 2:
            effective_type_list.append(type)
        elif value == 4:
            extreme_type_list.append(type)
        elif value == 1 / 2:
            resistant_type_list.append(type)
        elif value == 1 / 4:
            extreme_resistant_type_list.append(type)
        elif value == 0:
            immune_type_list.append(type)
        else:  # value == 1
            continue

    if constraint_type_list:
        extreme_type_list = list(set(extreme_type_list).intersection(set(constraint_type_list)))
        effective_type_list = list(set(effective_type_list).intersection(set(constraint_type_list)))
        resistant_type_list = list(set(resistant_type_list).intersection(set(constraint_type_list)))
        extreme_resistant_type_list = list(set(extreme_resistant_type_list).intersection(set(constraint_type_list)))
        immune_type_list = list(set(immune_type_list).intersection(set(constraint_type_list)))

    return (list(map(lambda x: x.capitalize(), extreme_type_list)),
           list(map(lambda x: x.capitalize(), effective_type_list)),
           list(map(lambda x: x.capitalize(), resistant_type_list)),
           list(map(lambda x: x.capitalize(), extreme_resistant_type_list)),
           list(map(lambda x: x.capitalize(), immune_type_list)))

def move_type_damage_wrapper(pokemon, type_chart, constraint_type_list=None):
    if pokemon is None:
        return ""
    type_1 = None
    type_2 = None
    if pokemon.type_1:
        type_1 = pokemon.type_1.name
        if pokemon.type_2:
            type_2 = pokemon.type_2.name

    move_type_damage_prompt = ""
    extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list)

    move_type_damage_prompt = ""
    if extreme_effective_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(extreme_effective_type_list) +
                                   f"-type attack is extremely-effective (4x damage) to {pokemon.species}.")

    if effective_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(effective_type_list) +
                                   f"-type attack is super-effective (2x damage) to {pokemon.species}.")

    if resistant_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(resistant_type_list) +
                                   f"-type attack is ineffective (0.5x damage) to {pokemon.species}.")

    if extreme_resistant_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(extreme_resistant_type_list) +
                                   f"-type attack is highly ineffective (0.25x damage) to {pokemon.species}.")

    if immune_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(immune_type_list) +
                                   f"-type attack is zero effect (0x damage) to {pokemon.species}.")

    return move_type_damage_prompt

class LocalVGCSim():
    def __init__(self, 
                 battle: DoubleBattle,
                 move_effect: Dict,
                 pokemon_move_dict: Dict,
                 ability_effect: Dict,
                 pokemon_ability_dict: Dict,
                 item_effect: Dict,
                 pokemon_item_dict: Dict,
                 gen: GenData,
                 _dynamax_disable: bool,
                 _strategy: str='',
                 format: str='gen9vgc2025regi',
                 prompt_translate: Callable=None
        ):
        self.battle = deepcopy(battle)
        self.move_effect = move_effect
        self.pokemon_move_dict = pokemon_move_dict
        self.ability_effect = ability_effect
        self.pokemon_ability_dict = pokemon_ability_dict
        self.item_effect = item_effect
        self.gen = gen
        self._dynamax_disable = _dynamax_disable
        self._tera_disable = False
        self.strategy = _strategy
        self.format = format
        self.prompt_translate = prompt_translate

        self.switch_set = set()

        self.SPEED_TIER_COEFICIENT = 0.1
        self.HP_FRACTION_COEFICIENT = 0.4
        
        # Use cached moves set data instead of loading file
        if get_cached_moves_set is not None:
            self.moves_set = get_cached_moves_set(self.format)
        else:
            self.moves_set = {}

    def _get_pokemon_tag(self, pokemon: Optional[Pokemon], player_role: str, slot: int) -> str:
        """Get the battle tag for a pokemon (e.g., 'p1a', 'p2b')"""
        if pokemon is None:
            return f"{player_role}{'a' if slot == 0 else 'b'}"
        # Try to find the pokemon in the team to get its identifier
        for ident, mon in self.battle.team.items():
            if mon == pokemon:
                return ident
        # Fallback
        return f"{player_role}{'a' if slot == 0 else 'b'}"

    def _get_opponent_tag(self, pokemon: Optional[Pokemon], opponent_role: str, slot: int) -> str:
        """Get the battle tag for opponent pokemon"""
        if pokemon is None:
            return f"{opponent_role}{'a' if slot == 0 else 'b'}"
        # Try to find in opponent team
        for ident, mon in self.battle.opponent_team.items():
            if mon == pokemon:
                return ident
        # Fallback
        return f"{opponent_role}{'a' if slot == 0 else 'b'}"

    def step(self, player_actions: DoubleBattleOrder, opponent_actions: DoubleBattleOrder):
        """
        Simulate one turn of a double battle.
        
        Args:
            player_actions: DoubleBattleOrder with first_order and second_order for player
            opponent_actions: DoubleBattleOrder with first_order and second_order for opponent
        """
        # Extract actions
        p1_action = player_actions.first_order if player_actions else None
        p2_action = player_actions.second_order if player_actions else None
        o1_action = opponent_actions.first_order if opponent_actions else None
        o2_action = opponent_actions.second_order if opponent_actions else None
        
        actions = [
            (p1_action, 0, True),   # player slot 0
            (p2_action, 1, True),   # player slot 1
            (o1_action, 0, False),  # opponent slot 0
            (o2_action, 1, False),  # opponent slot 1
        ]
        
        # Determine player and opponent roles
        player_role = self.battle.player_role
        opponent_role = 'p2' if player_role == 'p1' else 'p1'
        
        msg_all = []
        
        # Handle switches first
        for action, slot, is_player in actions:
            if action is None:
                continue
                
            m = action.order
            action_msg = action.message
            
            if 'switch' in action_msg and 'move' not in action_msg:
                action_name = action_msg.split(' ')[-1].title()
                health = 100
                
                if is_player:
                    for avail_mon in self.battle.available_switches[slot]:
                        if avail_mon.species.lower().replace(' ', '').replace('-', '') == action_name.lower().replace(' ', '').replace('-', ''):
                            health = int(avail_mon.current_hp_fraction * 100)
                            break
                    tag = self._get_pokemon_tag(None, player_role, slot)
                    msg = ['', 'switch', f'{tag}: {action_name}', '', f'{health}/100']
                else:
                    for avail_mon in self.battle.opponent_team.values():
                        if avail_mon.species.lower().replace(' ', '').replace('-', '') == action_name.lower().replace(' ', '').replace('-', ''):
                            health = int(avail_mon.current_hp_fraction * 100)
                            break
                    tag = self._get_opponent_tag(None, opponent_role, slot)
                    msg = ['', 'switch', f'{tag}: {action_name}', '', f'{health}/100']
                
                msg_all.append(msg)
        
        # Process switches
        for request in msg_all:
            self._handle_battle_message(request)
        
        msg_all = []
        
        # Get active pokemon after switches
        player_active = self.battle.active_pokemon
        opponent_active = self.battle.opponent_active_pokemon
        
        # Calculate speeds and priorities for all pokemon
        pokemon_data = []
        for i, (action, slot, is_player) in enumerate(actions):
            if action is None:
                continue
                
            mon = player_active[slot] if is_player else opponent_active[slot]
            if mon is None or mon.fainted:
                continue
                
            move = action.order if isinstance(action.order, Move) else None
            
            stats = mon.calculate_stats(battle_format=self.format)
            boosts = mon._boosts
            speed = round(stats['spe'] * self.boost_multiplier('spe', boosts['spe'])) * self.apply_protosynthesis(mon, 'spe')
            priority = move.priority if move and move.priority else 0
            
            pokemon_data.append({
                'index': i,
                'action': action,
                'slot': slot,
                'is_player': is_player,
                'pokemon': mon,
                'move': move,
                'speed': speed,
                'priority': priority,
                'stats': stats,
                'boosts': boosts
            })
        
        # Sort by priority (higher first), then speed (higher first)
        pokemon_data.sort(key=lambda x: (-x['priority'], -x['speed']))
        
        # Process moves in speed order
        for pdata in pokemon_data:
            action = pdata['action']
            slot = pdata['slot']
            is_player = pdata['is_player']
            mon = pdata['pokemon']
            move = pdata['move']
            
            if move is None:
                continue
            
            action_name = action.message.split(' ')[-1].title()
            move_target = action.move_target
            
            # Determine target(s) for the move
            targets = []
            if move_target == DoubleBattle.EMPTY_TARGET_POSITION:
                # Default targeting or spread move
                # Check if it's a spread move (hits both opponents)
                spread_moves = {'earthquake', 'surf', 'discharge', 'rock slide', 'blizzard', 'heat wave'}
                if move.id.lower() in spread_moves or move.target == 'allAdjacentFoes':
                    # Spread move - hits both opponents
                    if is_player:
                        targets = [(opponent_active[0], 0), (opponent_active[1], 1)]
                    else:
                        targets = [(player_active[0], 0), (player_active[1], 1)]
                else:
                    # Single target - default to first opponent
                    if is_player:
                        target_mon = opponent_active[0] if opponent_active[0] and not opponent_active[0].fainted else opponent_active[1]
                        if target_mon:
                            targets = [(target_mon, 0 if target_mon == opponent_active[0] else 1)]
                    else:
                        target_mon = player_active[0] if player_active[0] and not player_active[0].fainted else player_active[1]
                        if target_mon:
                            targets = [(target_mon, 0 if target_mon == player_active[0] else 1)]
            else:
                # Explicit target
                if move_target == 1:  # Opponent slot 1 (left)
                    if is_player:
                        if opponent_active[0] and not opponent_active[0].fainted:
                            targets = [(opponent_active[0], 0)]
                    else:
                        if player_active[0] and not player_active[0].fainted:
                            targets = [(player_active[0], 0)]
                elif move_target == 2:  # Opponent slot 2 (right)
                    if is_player:
                        if opponent_active[1] and not opponent_active[1].fainted:
                            targets = [(opponent_active[1], 1)]
                    else:
                        if player_active[1] and not player_active[1].fainted:
                            targets = [(player_active[1], 1)]
            
            # Calculate damage for each target
            for target_mon, target_slot in targets:
                if target_mon is None or target_mon.fainted:
                    continue
                
                # Calculate damage
                if move.category != MoveCategory.STATUS:
                    base_dmg = self.calc_base_dmg(mon, target_mon, move, boosts1=pdata['boosts'], 
                                                  boosts2=target_mon._boosts, team=self.battle.team if is_player else self.battle.opponent_team)
                    # Apply spread modifier if multi-target
                    if len(targets) > 1:
                        base_dmg *= 0.75  # Spread moves do 75% damage in doubles
                    
                    # Modify damage
                    dmg = self.modify_damage(base_dmg, mon, target_mon, move, None)
                    
                    # Apply damage
                    stats_target = target_mon.calculate_stats(battle_format=self.format)
                    hp_total = stats_target['hp']
                    hp_current = target_mon.current_hp_fraction * hp_total
                    hp_new = max(hp_current - dmg, 0)
                    hp_percent = int((hp_new / hp_total) * 100) if hp_total > 0 else 0
                    
                    # Create messages
                    if is_player:
                        player_tag = self._get_pokemon_tag(mon, player_role, slot)
                        opp_tag = self._get_opponent_tag(target_mon, opponent_role, target_slot)
                    else:
                        player_tag = self._get_opponent_tag(mon, opponent_role, slot)
                        opp_tag = self._get_pokemon_tag(target_mon, player_role, target_slot)
                    
                    msg = ['', 'move', f'{player_tag}: {mon.species.title()}', f'{action_name}', f'{opp_tag}: {target_mon.species.title()}']
                    msg_all.append(msg)
                    
                    if hp_percent == 0:
                        msg = ['', '-damage', f'{opp_tag}: {target_mon.species.title()}', '0 fnt']
                    else:
                        msg = ['', '-damage', f'{opp_tag}: {target_mon.species.title()}', f'{hp_percent}/100']
                    msg_all.append(msg)
                    
                    # Update battle state
                    pokemon_str, hp_status = msg[2:4]
                    self.battle.get_pokemon(pokemon_str).damage(hp_status)
                    
                    # Handle status effects
                    if move.status is not None and hp_percent > 0:
                        msg = ['', '-status', f'{opp_tag}: {target_mon.species.title()}', f'{move.status}']
                        msg_all.append(msg)
                else:
                    # Status move
                    if is_player:
                        player_tag = self._get_pokemon_tag(mon, player_role, slot)
                        opp_tag = self._get_opponent_tag(target_mon, opponent_role, target_slot)
                    else:
                        player_tag = self._get_opponent_tag(mon, opponent_role, slot)
                        opp_tag = self._get_pokemon_tag(target_mon, player_role, target_slot)
                    
                    msg = ['', 'move', f'{player_tag}: {mon.species.title()}', f'{action_name}', f'{opp_tag}: {target_mon.species.title()}']
                    msg_all.append(msg)
                    
                    if move.status is not None:
                        msg = ['', '-status', f'{opp_tag}: {target_mon.species.title()}', f'{move.status}']
                        msg_all.append(msg)
        
        # Process all messages
        for request in msg_all:
            self._handle_battle_message(request)
        
        return

    def get_llm_system_prompt(self, _format: str, llm: GPTPlayer | LLAMAPlayer = None, team_str: str=None, model: str='gpt-4o'):
        # sleep to make sure server has sent pokemon team information first
        if 'random' in _format:
            if llm is not None:
                sleep(1)
                strategy_prompt = f""
                for poke_str in self.battle.team.keys():
                    mon = self.battle.team[poke_str]
                    strategy_prompt += f"{mon.species}"
                    try:
                        if mon.item: strategy_prompt += f" @ {self.item_effect[mon.item]['name']}"
                    except:
                        pass
                    strategy_prompt += '\n'
                    if mon.ability: strategy_prompt += f"Ability: {mon.ability}\n"
                    strategy_prompt += "EVs: 85 HP/ 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe\n"
                    strategy_prompt += f"Bashful Nature\n"
                    for move in mon.moves.values():
                        strategy_prompt += f'- {move.id}\n'
                    strategy_prompt += '\n'

                strategy_prompt += "\nI play competitive Pokémon battles. How do I play this teams effectively?"
                self.strategy, _ = llm.get_LLM_query("", strategy_prompt, max_tokens=1000, model=model)
        elif team_str != None and llm is not None:
            strategy_prompt = team_str
            strategy_prompt += "\n\nI play competitive Pokémon battles. How do I play this team effectively?"
            self.strategy, _ = llm.get_LLM_query("", strategy_prompt, max_tokens=1000, model=model)
        return self.strategy

    def get_hp_diff(self):
        # calculate expected hp difference between p1 and p2
        hp_diff = 0.

        for mon in self.battle.team.values():
            hp_diff += mon.current_hp_fraction

        for mon in self.battle.opponent_team.values():
            hp_diff -= mon.current_hp_fraction
        remaining_pokemon = 6 - len(self.battle.opponent_team.values())
        hp_diff -= remaining_pokemon

        return hp_diff

    def get_player_prompt(self, return_actions=False, return_choices=False, idx=0):
        # For doubles, idx specifies which active pokemon (0 or 1)
        if return_actions:
            system_prompt, state_prompt, state_action_prompt, action_prompt_switch, action_prompt_move = self.prompt_translate(self, self.battle, return_actions=return_actions, idx=idx)
        elif return_choices:
            system_prompt, state_prompt, state_action_prompt, action_choice_switch, action_choice_move = self.prompt_translate(self, self.battle, return_choices=return_choices, idx=idx)
        else:
            system_prompt, state_prompt, state_action_prompt = self.prompt_translate(self, self.battle, idx=idx)

        # Check if pokemon at idx is fainted or has no moves
        active_mon = self.battle.active_pokemon[idx] if idx < len(self.battle.active_pokemon) else None
        if active_mon is None or active_mon.fainted or len(self.battle.available_moves[idx]) == 0:
            constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
        elif len(self.battle.available_switches[idx]) == 0:
            constraint_prompt_io = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>", "target":"<target_number>"}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>", "target":"<target_number>"} or {"thought":"<step-by-step-thinking>"}\n'''
        else:
            constraint_prompt_io = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>", "target":"<target_number>"} or {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>", "target":"<target_number>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
        if return_actions:
            return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move
        elif return_choices:
            return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_choice_switch, action_choice_move
        return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt

    def get_opponent_prompt(self, state_prompt, return_actions=False):
        system_prompt = (
                "You are a pokemon battler that targets to win the pokemon battle by predicting the action that the opposing battler will use. Your opponent can choose to take a move or switch in another pokemon. Here are some battle tips:"
                " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
                " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
                " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
                " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
                " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
                )
        system_prompt += ' Output the action the opposing battler will use. \
            You may not be able to observe all possible moves and pokemon that the opposing battler will use so you will need to both select the action based on their move history \
            and any possible unseen moves/pokemon. '
        state_switch_prompt = ''
        
        # Get turn summary (need to implement get_turn_summary or use state_prompt)
        battle_prompt = state_prompt  # Simplified for now

        # get viewable opponent pokemon
        observable_switches = []
        opponent_fainted_num = 0
        for _, opponent_pokemon in self.battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1
            elif not opponent_pokemon.active:
                observable_switches.append(opponent_pokemon.species)
        opponent_unfainted_num = 6 - opponent_fainted_num
        swich_prompt = ''
        if opponent_unfainted_num > 1:
            state_switch_prompt += f'The opponent may switch to any of their remaining {opponent_unfainted_num} pokemon.\n'
        if len(observable_switches) > 0:
            swich_prompt += f'[<opponent_switch_pokemon_name>] = {observable_switches}\n'
        
        # get definite moves for the current pokemon (use first active pokemon for now)
        opponent_moves = []
        if len(self.battle.opponent_active_pokemon) > 0 and self.battle.opponent_active_pokemon[0] is not None:
            opp_mon = self.battle.opponent_active_pokemon[0]
            if opp_mon.moves:
                opponent_moves = [move.id for move in opp_mon.moves.values()]
        
        action_prompt = f' Opponent\'s current Pokemon: {opp_mon.species if len(self.battle.opponent_active_pokemon) > 0 and self.battle.opponent_active_pokemon[0] else "Unknown"}.\nChoose only from the following opponent action choices:\n'
        move_prompt = ''
        if len(opponent_moves) > 0:
            move_prompt += f"[<opponent_move_name>] = {opponent_moves}\n"

        if len(self.battle.opponent_active_pokemon) == 0 or (self.battle.opponent_active_pokemon[0] is not None and self.battle.opponent_active_pokemon[0].fainted):
            constraint_prompt_io = '''Choose the most suitable opponent pokemon to switch. Your output MUST be a JSON like: {"switch":"<opponent_switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable opponent pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<opponent_step-by-step-thinking>", "switch":"<opponent_switch_pokemon_name>"}\n'''
        elif opponent_unfainted_num == 1:
            constraint_prompt_io = '''Choose the best opponent action and your output MUST be a JSON like: {"move":"<opponent_move_name>"}\n'''
            constraint_prompt_cot = '''Choose the best opponent action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<opponent_step-by-step-thinking>", "move":"<opponent_move_name>"} or {"thought":"<opponent_step-by-step-thinking>"}\n'''
        else:
            constraint_prompt_io = '''Choose the best opponent action and your output MUST be a JSON like: {"move":"<opponent_move_name>"} or {"switch":"<opponent_switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the best opponent action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<opponent_step-by-step-thinking>", "move":"<opponent_move_name>"} or {"thought":"<opponent_step-by-step-thinking>", "switch":"<opponent_switch_pokemon_name>"}\n'''

        state_action_prompt = battle_prompt + action_prompt + move_prompt + state_switch_prompt + swich_prompt
        if return_actions:
            return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, move_prompt, swich_prompt
        return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt

    def is_terminal(self):
        return self.battle._finished

    def check_status(self, status):
        if status:
            if status.value == 1:
                return "burnt"
            elif status.value == 2:
                return "fainted"
            elif status.value == 3:
                return "frozen"
            elif status.value == 4:
                return "paralyzed"
            elif status.value == 5:
                return "poisoned"
            elif status.value == 7:
                return "toxic"
            elif status.value == 6:
                return "sleeping"
        else:
            return ""

    def boost_multiplier(self, state, level):
        if state == "accuracy":
            if level == 0:
                return 1.0
            if level == 1:
                return 1.33
            if level == 2:
                return 1.66
            if level == 3:
                return 2.0
            if level == 4:
                return 2.5
            if level == 5:
                return 2.66
            if level == 6:
                return 3.0
            if level == -1:
                return 0.75
            if level == -2:
                return 0.6
            if level == -3:
                return 0.5
            if level == -4:
                return 0.43
            if level == -5:
                return 0.36
            if level == -6:
                return 0.33
        else:
            if level == 0:
                return 1.0
            if level == 1:
                return 1.5
            if level == 2:
                return 2.0
            if level == 3:
                return 2.5
            if level == 4:
                return 3.0
            if level == 5:
                return 3.5
            if level == 6:
                return 4.0
            if level == -1:
                return 0.67
            if level == -2:
                return 0.5
            if level == -3:
                return 0.4
            if level == -4:
                return 0.33
            if level == -5:
                return 0.29
            if level == -6:
                return 0.25
        raise ValueError(f'Boost level not found {state} {level}')

    def apply_protosynthesis(self, mon: Pokemon, returned_stat):
        if (mon.ability == 'protosynthesis' or mon.ability == 'quarkdrive') and mon.item == 'boosterdrive':
            best_stat_val = mon.stats['atk']
            best_stat = 'atk'
            for stat in ['def', 'spa', 'spd', 'spe']:
                if best_stat_val < mon.stats[stat]:
                    best_stat_val = mon.stats[stat]
                    best_stat = stat
            if best_stat == returned_stat:
                if best_stat == 'spe':
                    return 1.5
                else:
                    return 1.3
        return 1.0

    def modify_base_power(self, mon: Pokemon, target: Pokemon, move: Move, team=None) -> float:
        power = move.base_power
        # weight based modifiers based on difference in health
        weight_moves_diff = {'heavyslam', 'heatcrash'}
        if weight_moves_diff.intersection([move.id]):
            relative_weight = target.weight / mon.weight
            if relative_weight < 0.2:
                power = 120
            elif relative_weight < 0.25:
                power = 100
            elif relative_weight < 0.334:
                power = 80
            elif relative_weight < 0.5:
                power = 60
            else:
                power = 40
        # weight based modifiers based on opponent's weight
        weight_moves_opp = {'grassknot', 'lowkick'}
        if weight_moves_opp.intersection([move.id]):
            if target.weight > 200:
                power = 120
            elif target.weight > 100:
                power = 100
            elif target.weight > 50:
                power = 80
            elif target.weight > 25:
                power = 60
            elif target.weight > 10:
                power = 40
            else:
                power = 20
        # ability based modifiers
        if mon.ability == 'technician' and power <= 60:
            power *= 1.5
        if move.id == 'acrobatics':
            if mon.item == None or mon.item == 'flyinggem':
                power *= 2
        if mon.ability == 'supremeoverlord' and team is not None:
            boost_atk = 0
            for teammate in team.values():
                if teammate.fainted:
                    boost_atk += 0.1
            power *= 1.0 + boost_atk
        return power
    
    def apply_item(self, mon: Pokemon, boosts: Dict[str, int]) -> Dict[str, int]:
        if boosts is None:
            return None
        boosts = boosts.copy()
        item = mon.item
        def boost(stat: str, amount: float):
                boosts[stat] += amount
                boosts[stat] = np.floor(boosts[stat])
                if boosts[stat] > 6:
                    boosts[stat] = 6
                elif boosts[stat] < -6:
                    boosts[stat] = -6
                return
        if item != None:
            if mon.status:
                if mon.ability == 'guts':
                    boost('atk', 1.5)
                elif mon.ability == 'quickfeet':
                    boost('spe', 1.5)
                elif mon.status == Status.BRN:
                    boost('atk', 0.5)
                elif mon.status == Status.PAR:
                    boost('spe', 0.5)
                if mon.ability == 'marvelscale':
                    boost('def', 1.5)
                    
            if mon.species == 'pikachu' and item == 'lightball':
                boost('atk', 2.0)
                boost('spa', 2.0)
            if mon.species in ['marowak', 'cubone'] and item == 'thickclub':
                boost('atk', 2.0)
            if mon.species == 'ditto':
                if item == 'quickpowder':
                    boost('spe', 2.0)
                elif item == 'metalpowder':
                    boost('def', 2.0)
            if mon.is_dynamaxed:
                boost('hp', 2.0)
            if item == 'choiceband':
                boost('atk', 1.5)
            elif item == 'choicespecs':
                boost('spa', 1.5)
            elif item == 'choicescarf':
                boost('spe', 1.5)
            elif item == 'assaultvest':
                boost('spd', 1.5)
            elif item == 'furcoat':
                boost('def', 2.0)
            elif mon.species == 'clamperl':
                if item == 'deepseatooth':
                    boost('spa', 2)
                elif item == 'deepseascale':
                    boost('spd', 2)
            elif item == 'ironball':
                boost('spe', 0.5)
                
        if mon.ability in ['purepower', 'hugepower']:
            boost('atk', 2.0)
        elif mon.ability == 'hustle' or (mon.ability == 'gorillatactics' and not mon.is_dynamaxed):
            boost('atk', 1.5)
        elif mon.ability == 'furcoat':
            boost('def', 2.0)
        elif mon.ability == 'defeatist' and mon.current_hp_fraction <= 0.5:
            boost('atk', 0.5)
            
        weather = self.battle.weather
        if weather:
            if weather == 'sand':
                if 'rock' in mon.types:
                    boost('spd', 1.5)
                if mon.ability == 'sandrush':
                    boost('spe', 2.0)
            if weather in ['hail', 'snow'] and mon.ability == 'slushrush':
                boost('spe', 2.0)
            if weather == 'snow' and 'ice' in mon.types:
                boost('def', 1.5)
            if item != 'utilityumbrella':
                if weather in ['sun', 'harshsunshine']:
                    if mon.ability == 'solarpower':
                        boost('spa', 1.5)
                    elif mon.ability == 'chlorophyll':
                        boost('spe', 1.5)
                    elif mon.ability == 'orichalcumpulse':
                        boost('atk', 1.3)
            if weather in ['rain', 'heavyrain'] and mon.ability == 'swiftswim':
                boost('spe', 2.0)
            boost('spa', 0.5)

        return boosts
    
    def calc_base_dmg(self, 
                      pokemon: Pokemon, 
                      target: Pokemon, 
                      move: Move,
                      boosts1: Dict[str, int]=None, 
                      boosts2: Dict[str, int]=None, 
                      team=None,
                      ) -> float:
        baseDamage = 2
        level = pokemon.level
        power = self.modify_base_power(pokemon, target, move, team)
        stats = pokemon.calculate_stats(battle_format=self.format)
        if boosts1 is None:
            boosts1 = pokemon._boosts
        active_boosts = self.apply_item(pokemon, boosts1)
        stats_target = target.calculate_stats(battle_format=self.format)
        if boosts2 is None:
            boosts2 = target._boosts
        target_boosts = self.apply_item(target, boosts2)
        A = -1
        D = -1
        if move.category == MoveCategory.PHYSICAL:
            A = stats['atk'] if active_boosts['atk']==0 else round(stats['atk']*self.boost_multiplier('atk', active_boosts['atk']))
            A = A * self.apply_protosynthesis(pokemon, 'atk')
            D = stats_target['def'] if target_boosts['def']==0 else round(stats_target['def']*self.boost_multiplier('def', target_boosts['def']))
            D = D * self.apply_protosynthesis(pokemon, 'def')
        elif move.category == MoveCategory.SPECIAL:
            A = stats['spa'] if active_boosts['spa']==0 else round(stats['spa']*self.boost_multiplier('spa', active_boosts['spa']))
            A = A * self.apply_protosynthesis(pokemon, 'spa')
            D = stats_target['spd'] if target_boosts['spd']==0 else round(stats_target['spd']*self.boost_multiplier('spd', target_boosts['spd']))
            D = D * self.apply_protosynthesis(pokemon, 'spd')
        assert A != -1
        baseDamage += (((2*level) / 5. + 2) * power * A / D) / 50. + 2
        return baseDamage

    def modify_damage(self, baseDamage: float, pokemon: Pokemon, target: Pokemon, move: Move, target_move: Move, use_expected: bool=True) -> float:
        if (not move.type):
            move.type = '???'
        type = move.type

        type_1 = pokemon.type_1.name
        type_2 = None
        if pokemon.type_2 != None:
            type_2 = pokemon.type_2.name
        # STAB
        stab = 1
        if (type != '???'):
            if type.name in [type_1, type_2]:
                stab = pokemon.stab_multiplier
                
        baseDamage *= stab

        # types
        opponent_type_list = []
        if target.terastallized:
            opponent_type_list.append(target._terastallized_type)
        else:
            if target.type_1:
                type_1 = target.type_1.name
                opponent_type_list.append(type_1)

                if target.type_2:
                    type_2 = target.type_2.name
                    opponent_type_list.append(type_2)

        extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
                                    type_1, type_2, self.gen.type_chart, [type.name])

        if extreme_effective_type_list:
            baseDamage *= 4
        elif effective_type_list:
            baseDamage *= 2
        elif resistant_type_list:
            baseDamage *= 0.5
        elif extreme_resistant_type_list:
            baseDamage *= 0.25
        elif immune_type_list:
            baseDamage *= 0

        # check for item immunity
        if target.item is not None:
            if target.item.lower() == 'airballoon':
                if type == 'ground':
                    baseDamage *= 0
        # ability immunity
        if target.ability == 'voltabsorb' and move.type == 'electric':
            baseDamage *= 0
        if (target.ability == 'waterabsorb' or target.ability == 'dryskin') and move.type == 'water':
            baseDamage *= 0
        if target.ability == 'levitate' and move.type == 'ground':
            baseDamage *= 0
        if target.ability == 'flashfire' and move.type == 'fire':
            baseDamage *= 0
        if target.ability == 'dryskin' and move.type == 'fire':
            baseDamage *= 1.25
        if target.ability == 'wonderguard' and not (extreme_effective_type_list or effective_type_list):
            baseDamage *= 0

        if (pokemon.status == Status.BRN and move.category == MoveCategory.PHYSICAL and not pokemon.ability == 'guts'):
            if (self.gen.gen < 6 or move.id != 'facade'):
                baseDamage *= 0.5

        if (self.gen.gen == 5 and baseDamage == 0): baseDamage = 1

        if pokemon.item == 'LifeOrb':
            baseDamage *= 1.3

        if target_move != None:
            if ((move.is_z or pokemon.is_dynamaxed) and target_move.id == 'protect'):
                baseDamage *= 0.25
            if target_move.category == MoveCategory.STATUS and (move.id == 'suckerpunch' or move.id == 'thunderclap'):
                baseDamage *= 0

        if (self.gen.gen != 5 and baseDamage == 0): baseDamage = 1

        if use_expected:
            baseDamage *= move.accuracy
        
        return int(baseDamage) * move.expected_hits

    def _handle_battle_message(self, split_message: List[str]):
        """Handles a battle message."""
        description = ''
        battle = self.battle
        if split_message[1] == "switch":
            self.switch_set.add(split_message[2])
            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[4])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[4]]

            description = " " + split_message[2].split(" ")[0] + " sent out " + split_message[2].split(": ")[-1] + "."
            description = description.replace("p2a:", "Player2").replace("p1a:", "Player1").replace("p2b:", "Player2").replace("p1b:", "Player1")
        elif split_message[1] == "turn":
            if len(battle.speed_list) == 2:
                description = f" {battle.speed_list[0]} outspeeded {battle.speed_list[1]} in this turn."
            description += "[sep]Turn " + split_message[2] + ":"
        elif split_message[1] == "drag":
            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[4])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[4]]

            description = " " + split_message[2] + "was dragged out."

        elif split_message[1] == "faint":
            description = " " + split_message[2] + " faint."

        elif split_message[1] == "move":
            description = " " + split_message[2] + " used "+ split_message[3] + "."
            battle.speed_list.append(split_message[2])

        elif split_message[1] == "cant":
            if split_message[3] == "frz":
                reason = "frozen"
            elif split_message[3] == "par":
                reason = "paralyzed"
            elif split_message[3] == "slp":
                reason = "sleeping"
            else:
                reason = split_message[3]

            description = " " + split_message[2] + " cannot move because of " + reason + "."

        elif split_message[1] == "-start":
            description = " " + split_message[2] + " started " + split_message[3] + "."
            if len(split_message) > 4:
                if split_message[4]:
                    description = " " + split_message[2] + " started " + split_message[3] + " due to " + split_message[4] + "."

        elif split_message[1] == "-end":
            description = " " + split_message[2] + " stop " + split_message[3] + "."

        elif split_message[1] == "-fieldstart":
            description = " Field start: " + split_message[2] + " ran across the battlefield."

        elif split_message[1] == "-fieldend":
            description = " Field end: " + split_message[2] + " disappeared from the battlefield."

        elif split_message[1] == "-ability":
            description = " " + split_message[2] + "'s ability: " + split_message[3] + "."

        elif split_message[1] == "-supereffective":
            description = " The move was super effective to " + split_message[2] + "."

        elif split_message[1] == "-resisted":
            description = " The move was ineffective to " + split_message[2] + "."

        elif split_message[1] == "-heal":
            try:
                previous_hp = battle.pokemon_hp_log_dict[split_message[2]][-1].split(" ")[0]
            except:
                previous_hp = "100/100"

            if previous_hp == "0":
                previous_hp_fraction = 0
            else:
                previous_hp_fraction = round(float(previous_hp.split("/")[0]) / float(previous_hp.split("/")[1]) * 100)

            current_hp = split_message[3].split(" ")[0]
            if current_hp == "0":
                current_hp_fraction = 0
            else:
                current_hp_fraction = round(float(current_hp.split("/")[0]) / float(current_hp.split("/")[1]) * 100)

            delta_hp_fraction = current_hp_fraction - previous_hp_fraction

            if len(split_message) > 4:
                description = f" {split_message[2]} restored {delta_hp_fraction}% of HP ({current_hp_fraction}% left) {split_message[4]}."
            else:
                description = f" {split_message[2]} restored {delta_hp_fraction}% of HP ({current_hp_fraction}% left)."
            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[3])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[3]]

        elif split_message[1] == "-damage":
            try:
                previous_hp = battle.pokemon_hp_log_dict[split_message[2]][-1].split(" ")[0]
            except:
                previous_hp = "100/100"

            if previous_hp == "0":
                previous_hp_fraction = 0
            else:
                previous_hp_fraction = round(float(previous_hp.split("/")[0]) / float(previous_hp.split("/")[1]) * 100)

            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[3])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[3]]

            current_hp = split_message[3].split(" ")[0]
            if current_hp == "0":
                current_hp_fraction = 0
            else:
                current_hp_fraction = round(float(current_hp.split("/")[0]) / float(current_hp.split("/")[1]) * 100)

            delta_hp_fraction = previous_hp_fraction - current_hp_fraction

            if "oroark" in split_message[2]:
                if len(split_message) > 4:
                    description = f" {split_message[2]}'s HP was damaged to {current_hp_fraction}% {split_message[4]}."
                else:
                    description = f" It damaged {split_message[2]}'s HP to {current_hp_fraction}%."
            else:
                if len(split_message) > 4:
                    description = f" {split_message[2]}'s HP was damaged by {delta_hp_fraction}% {split_message[4]} ({current_hp_fraction}% left)."
                else:
                    description = f" It damaged {split_message[2]}'s HP by {delta_hp_fraction}% ({current_hp_fraction}% left)."

        elif split_message[1] == "-unboost":
            description = " It decreased " + split_message[2] + "'s " + split_message[3] + " " + split_message[4] + " level."

        elif split_message[1] == "-boost":
            description = " It boosted " + split_message[2] + "'s " + split_message[3] + " " + split_message[4] + " level."

        elif split_message[1] == "-fail":
            description = " But it failed."

        elif split_message[1] == "-miss":
            description = " It missed."

        elif split_message[1] == "-activate":
            description = " " + split_message[2] + " activated " + split_message[3] + "."

        elif split_message[1] == "-immune":
            description = f" but had zero effect to {split_message[2]}."

        elif split_message[1] == "-crit":
            description = " A critical hit."

        elif split_message[1] == "-status":
            status_dict = {"brn": "burnt", "frz": "frozen", "par": "paralyzed", "slp": "sleeping", "tox": "toxic", "psn": "poisoned"}
            description = " It caused " + split_message[2] + " " + status_dict[split_message[3]] + "."
        
        if description:
            battle.battle_msg_history = battle.battle_msg_history + description

        self.battle.parse_message(split_message)


class VGCSimNode():
    def __init__(self, 
                 battle: DoubleBattle, 
                 move_effect,
                 pokemon_move_dict,
                 ability_effect,
                 pokemon_ability_dict,
                 item_effect,
                 pokemon_item_dict,
                 gen,
                 _dynamax_disable,
                 depth: int=0,
                 format='gen9vgc2025regi',
                 prompt_translate=None,
                 sim=None,
                ):
        if sim is None:
            self.simulation = LocalVGCSim(battle, 
                                    move_effect,
                                    pokemon_move_dict,
                                    ability_effect,
                                    pokemon_ability_dict,
                                    item_effect,
                                    pokemon_item_dict,
                                    gen,
                                    _dynamax_disable,
                                    format=format,
                                    prompt_translate=prompt_translate,
                                    )
        else:
            self.simulation = sim
        self.depth = depth
        self.action = None  # DoubleBattleOrder for player
        self.action_opp: DoubleBattleOrder = None  # DoubleBattleOrder for opponent
        self.parent_node = None
        self.parent_action = None
        self.hp_diff = 0
        self.children: List['VGCSimNode'] = []

