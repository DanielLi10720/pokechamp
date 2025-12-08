import ast
import asyncio
from copy import copy, deepcopy
import datetime
import json
import os
import random
import sys

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import Player, BattleOrder, DoubleBattleOrder
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.concurrency import POKE_LOOP
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from poke_env.environment.move import Move
import time
import json
from poke_env.data.gen_data import GenData
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer
from pokechamp.openrouter_player import OpenRouterPlayer
from pokechamp.gemini_player import GeminiPlayer
from pokechamp.ollama_player import OllamaPlayer
from pokechamp.data_cache import (
    get_cached_move_effect,
    get_cached_pokemon_move_dict,
    get_cached_ability_effect,
    get_cached_pokemon_ability_dict,
    get_cached_item_effect,
    get_cached_pokemon_item_dict,
    get_cached_pokedex
)
from pokechamp.minimax_optimizer import (
    get_minimax_optimizer,
    initialize_minimax_optimization,
    fast_battle_evaluation,
    create_battle_state_hash,
    OptimizedSimNode
)
from poke_env.player.local_simulation import LocalSim, SimNode
from poke_env.player.local_vgc_simulation import LocalVGCSim, VGCSimNode
from difflib import get_close_matches
from pokechamp.prompts import get_number_turns_faint, get_status_num_turns_fnt, state_translate, get_gimmick_motivation

DEBUG=True

class LLMVGCPlayer(Player):
    def __init__(self,
                 battle_format="gen9vgc2025regi",
                 api_key="",
                 backend="gpt-4-1106-preview",
                 temperature=1.0,
                 prompt_algo="io",
                 log_dir=None,
                 team=None,
                 save_replays=None,
                 account_configuration=None,
                 server_configuration=None,
                 K=2,
                 _use_strat_prompt=False,
                 prompt_translate: Callable=state_translate,
                 device=0,
                 llm_backend=None
                 ):

        super().__init__(battle_format=battle_format,
                         team=team,
                         save_replays=save_replays,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate

        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt
        
        # Use cached data instead of loading files repeatedly
        self.move_effect = get_cached_move_effect()
        # only used in old prompting method, replaced by statistcal sets data
        self.pokemon_move_dict = get_cached_pokemon_move_dict()
        self.ability_effect = get_cached_ability_effect()
        # only used is old prompting method
        self.pokemon_ability_dict = get_cached_pokemon_ability_dict()
        self.item_effect = get_cached_item_effect()
        # unused
        # with open(f"./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
        #     self.pokemon_item_dict = json.load(f)
        self.pokemon_item_dict = get_cached_pokemon_item_dict()
        self._pokemon_dict = get_cached_pokedex(self.gen.gen)

        self.last_plan = ""

        if llm_backend is None:
            print(f"Initializing backend: {backend}")  # Debug logging
            if backend.startswith('ollama/'):
                # Ollama models - extract model name after 'ollama/'
                model_name = backend.replace('ollama/', '')
                print(f"Using Ollama with model: {model_name}")
                self.llm = OllamaPlayer(model=model_name, device=device)
            elif 'gpt' in backend and not backend.startswith('openai/'):
                self.llm = GPTPlayer(self.api_key)
            elif 'llama' == backend:
                self.llm = LLAMAPlayer(device=device)
            elif 'gemini' in backend:
                self.llm = GeminiPlayer(self.api_key)
            elif backend.startswith(('openai/', 'anthropic/', 'google/', 'meta/', 'mistral/', 'cohere/', 'perplexity/', 'deepseek/', 'microsoft/', 'nvidia/', 'huggingface/', 'together/', 'replicate/', 'fireworks/', 'localai/', 'vllm/', 'sagemaker/', 'vertex/', 'bedrock/', 'azure/', 'custom/')):
                # OpenRouter supports hundreds of models from various providers
                self.llm = OpenRouterPlayer(self.api_key)
            else:
                raise NotImplementedError('LLM type not implemented:', backend)
        else:
            self.llm = llm_backend
        self.llm_value = self.llm
        self.K = K      # for minimax, SC, ToT
        self.use_optimized_minimax = True  # Enable optimized minimax by default
        self._minimax_initialized = False
        # Configuration for time optimization
        self.use_damage_calc_early_exit = True  # Use damage calculator to exit early when advantageous
        self.use_llm_value_function = True  # Use LLM for leaf node evaluation (vs fast heuristic)
        self.max_depth_for_llm_eval = 2  # Only use LLM evaluation for shallow depths to save time
    
    def _send_thinking_message(self, battle: AbstractBattle, message: str):
        """
        Send LLM thinking as chat messages during battle in 1000-character chunks.
        Based on TimeoutLLMPlayer._send_chat_message implementation.
        """
        try:
            # Split message into 1000-character chunks
            max_chunk_size = 950  # Leave room for turn prefix
            chunks = []
            
            for i in range(0, len(message), max_chunk_size):
                chunk = message[i:i + max_chunk_size]
                chunks.append(chunk)
            
            # Create an async function to send all message chunks
            async def send_message_async():
                try:
                    print(f"   Sending thinking to battle chat ({len(chunks)} parts)...")
                    
                    for part_num, chunk in enumerate(chunks, 1):
                        if len(chunks) == 1:
                            # Single message
                            chat_message = f"Turn #{battle.turn} thinking: {chunk}"
                        else:
                            # Multiple parts
                            chat_message = f"Turn #{battle.turn} thinking ({part_num}/{len(chunks)}): {chunk}"
                        
                        await self.ps_client.send_message(chat_message, room=battle.battle_tag)
                        
                        # Small delay between multiple messages
                        if part_num < len(chunks):
                            await asyncio.sleep(0.15)
                    
                    # Send fast mode command once after all thinking
                    # await self.ps_client.send_message("/timer off", room=battle.battle_tag)
                    print(f"All thinking sent to {battle.battle_tag}")
                    
                except Exception as e:
                    print(f"Failed to send thinking message: {e}")
            
            # Submit to the poke loop for execution
            try:
                future = asyncio.run_coroutine_threadsafe(send_message_async(), POKE_LOOP)
                # Don't wait for completion to avoid blocking
            except Exception as e:
                print(f"   Could not schedule thinking message: {e}")
                
        except Exception as e:
            print(f"Failed to send thinking message: {e}")

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, llm=None, battle=None) -> str:
        if llm is None:
            output, _, raw_message = self.llm.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions, battle=battle, ps_client=self.ps_client)
        else:
            output, _, raw_message = llm.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions, battle=battle, ps_client=self.ps_client)
        
        # Send thinking message if battle is provided
        if battle is not None and raw_message and hasattr(self, 'ps_client') and self.ps_client:
            try:
                self._send_thinking_message(battle, raw_message)
            except Exception as e:
                print(f"Failed to send thinking message: {e}")
        
        return output
    
    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8)
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        pokemon = Pokemon(species=pokemon_str, gen=self.genNum)
        return pokemon

    def _parse_target_string(self, target_str: str) -> int:
        """
        Parse various word targets into proper integer values.
        
        Target mapping:
        - -2: Ally position 2 (in triples)
        - -1: Ally position 1 (self in doubles)
        - 0: EMPTY_TARGET_POSITION (no specific target, affects field/all)
        - 1: OPPONENT_1_POSITION (left opponent)
        - 2: OPPONENT_2_POSITION (right opponent)
        """
        target_str = target_str.lower().strip()
        
        # Self-targeting moves
        if target_str in ["self", "user", "myself", "own"]:
            return -1
        
        # Field/area effects (no specific target)
        if target_str in ["all", "alladjacent", "alladjacentfoes", "allies", "allyside", 
                         "allyteam", "foeside", "randomnormal", "scripted", "empty", "none", "0"]:
            return 0
        
        # Opponent targeting
        if target_str in ["opponent", "opponent1", "left", "leftopponent", "foe1", "1"]:
            return 1
        if target_str in ["opponent2", "right", "rightopponent", "foe2", "2"]:
            return 2
        
        # Ally targeting
        if target_str in ["ally", "ally1", "teammate", "partner", "-1"]:
            return -1
        if target_str in ["ally2", "teammate2", "partner2", "-2"]:
            return -2
        
        # Adjacent targeting (can target either opponent)
        if target_str in ["adjacent", "adjacentfoe", "normal", "any", "foe"]:
            return 1  # Default to first opponent
        
        # Try to parse as integer
        try:
            parsed = int(target_str)
            if parsed in [-2, -1, 0, 1, 2]:
                return parsed
        except ValueError:
            pass
        
        # Default fallback
        print(f"WARNING: Unknown target string '{target_str}', using default target")
        return 0

    def parse_request(self, request: Dict[str, Any]) -> None:
        """
        Override parse_request to store team data for teampreview.
        """
        # Call parent parse_request first
        super().parse_request(request)
        
        # Store team data if this is a teampreview request
        if request.get("teamPreview", False) and "side" in request:
            self._teampreview_team_data = request["side"]["pokemon"]
            print(f"Stored teampreview team data: {len(self._teampreview_team_data)} Pokemon")
    
    # returns a double battle order object
    # within this method is a for loop that handles the decision for each active pokemon
    def choose_move(self, battle: AbstractBattle):
        sim = LocalVGCSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    self.strategy_prompt,
                    format=self.format,
                    prompt_translate=self.prompt_translate
        )
        next_action: List[Optional[BattleOrder]] = [None, None]
        if battle.turn <=1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(self.format, self.llm, team_str=self.team_str, model='gpt-4o-2024-05-13')
        
        # handle one choice paths for each active pokemon
        for i, mon in enumerate(battle.active_pokemon):
            if (mon is None or mon.fainted or battle.force_switch[i]) and len(battle.available_switches[i]) == 1:
                next_action[i] = BattleOrder(battle.available_switches[i][0])
            elif not (mon is None or mon.fainted) and len(battle.available_moves[i]) == 1 and len(battle.available_switches[i]) == 0:
                next_action[i] = self.choose_max_damage_move(battle, i)

        # handle all forced switch cases
        special_case_handled = False
        if all(battle.force_switch):
            #print("INFO: Both slots are forced to switch")
            # Check if we have a shared switch scenario (both forced to switch, limited options)
            total_available_switches = set()
            for idx in range(len(battle.active_pokemon)):
                if battle.force_switch[idx]:
                    for pokemon in battle.available_switches[idx]:
                        total_available_switches.add(pokemon.species)
            
            # If we have fewer total unique switches than forced slots, we need special handling
            if len(total_available_switches) < sum(battle.force_switch):
                print(f"WARNING: Both slots forced to switch but only {len(total_available_switches)} unique switches available")
                # Assign the first available switch to the first slot, None to the second
                next_action[0] = BattleOrder(battle.available_switches[0][0])
                next_action[1] = None
                special_case_handled = True
            else:
                # Normal case: enough switches for all slots
                # Ensure we don't try to use moves for any slot
                for idx in range(len(battle.active_pokemon)):
                    if not battle.force_switch[idx]:
                        next_action[idx] = None
        # for minimax, we will decide the move for both pokemon at the same
        if self.prompt_algo == "minimax":
            try:
                return self.vgc_tree_search(2, battle)
            except Exception as e:
                print(f'minimax step failed ({e}). Using dmg calc')
                print(f'Exception: {e}', 'passed')
                double_battle_order, turns = self.dmg_calc_move(battle)
                return double_battle_order

        for idx, mon in enumerate(battle.active_pokemon):
            # Skip individual processing if we already handled the special case
            if special_case_handled and battle.force_switch[idx]:
                continue
                
            # if force switch is true for any pokemon, but the current pokemon is not forced to switch, we need its action to be None
            # we will handle the forced to switch state in state_translate3
            if any(battle.force_switch):
                if not battle.force_switch[idx]:
                    next_action[idx] = None
                    continue
            
            # SAFEGUARD 1: Handle forced switch scenarios
            if battle.force_switch[idx]:
                # Only allow switches, no moves when forced to switch
                if len(battle.available_switches[idx]) == 0:
                    # No switches available - this shouldn't happen but handle gracefully
                    print(f"WARNING: Forced to switch but no switches available for slot {idx}")
                    next_action[idx] = None
                    continue
                
                # Build switch list excluding already chosen switches
                already_chosen = []
                for i, action in enumerate(next_action):
                    if i != idx and action is not None and not isinstance(action, DefaultBattleOrder):
                        if hasattr(action, 'order') and isinstance(action.order, Pokemon):
                            already_chosen.append(action.order.species)
                        elif hasattr(action, 'order') and hasattr(action.order, 'species'):
                            already_chosen.append(action.order.species)
                
                switches = [
                    pokemon.species
                    for pokemon in battle.available_switches[idx]
                    if pokemon.species not in already_chosen
                ]
                
                #print(f"DEBUG: Slot {idx} - Already chosen: {already_chosen}, Available: {[p.species for p in battle.available_switches[idx]]}, Filtered: {switches}")
                
                # If no valid switches left, use first available
                if not switches:
                    switches = [pokemon.species for pokemon in battle.available_switches[idx]]
                
                actions = [[], switches]  # No moves allowed when forced to switch

                constraint_prompt_io = f'''You MUST switch. Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {{"switch":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                constraint_prompt_cot = f'''You MUST switch. Choose the most suitable pokemon to switch by thinking step by step.Your thought should no more than 4 sentences. Your output MUST be a JSON like: {{"switch":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                constraint_prompt_tot_1 = '''You MUST switch. Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                constraint_prompt_tot_2 = '''You MUST switch. Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
                system_prompt, state_prompt, state_action_prompt = sim.prompt_translate(sim, battle, next_action=next_action, idx=idx)
                state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io 
                state_prompt_cot = state_prompt + state_action_prompt + constraint_prompt_cot
                state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_prompt_tot_1
                state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_prompt_tot_2
                retries = 10
                if self.prompt_algo == "io":
                    next_action[idx] = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions, idx=idx)
                if self.prompt_algo == "sc":
                    next_action[idx] = self.sc(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions, idx=idx)
                if self.prompt_algo == "tot":
                    next_action[idx] = self.tot(retries, system_prompt, state_prompt_tot_1, state_prompt_tot_2, state_action_prompt, battle, sim, actions=actions, idx=idx)
                # SAFEGUARD 2: Validate that the chosen switch is valid and not duplicate
                if next_action[idx] is not None and not isinstance(next_action[idx], DefaultBattleOrder):
                    if hasattr(next_action[idx], 'order') and isinstance(next_action[idx].order, Pokemon):
                        chosen_species = next_action[idx].order.species
                        
                        # Check if this species is already chosen by another slot
                        is_duplicate = False
                        for i, action in enumerate(next_action):
                            if i != idx and action is not None and not isinstance(action, DefaultBattleOrder):
                                if hasattr(action, 'order') and isinstance(action.order, Pokemon):
                                    if action.order.species == chosen_species:
                                        is_duplicate = True
                                        break
                        
                        if is_duplicate:
                            print(f"WARNING: LLM chose duplicate switch {chosen_species}, using fallback")
                            # Use first available non-duplicate switch
                            for pokemon in battle.available_switches[idx]:
                                if pokemon.species not in already_chosen:
                                    next_action[idx] = self.create_order(pokemon)
                                    break
                            else:
                                # If all switches are duplicates
                                next_action[idx] = self.create_order(battle.available_switches[idx][0])
                        elif chosen_species not in switches:
                            #print(f"WARNING: LLM chose invalid switch {chosen_species}, falling back to first available")
                            next_action[idx] = self.create_order(battle.available_switches[idx][0])
                    else:
                        #print(f"WARNING: Invalid action type for forced switch, setting to None")
                        next_action[idx] = None
                else:
                    # Fallback to first available switch
                    next_action[idx] = self.create_order(battle.available_switches[idx][0])
                
                continue
            
            system_prompt, state_prompt, state_action_prompt = sim.prompt_translate(sim, battle, next_action=next_action, idx=idx) # add lower case
            moves = [move.id for move in battle.available_moves[idx]]
            # switches = [pokemon.species for pokemon in battle.available_switches[idx]]
            # Exclude pokemon that are already chosen as switch-ins in next_action
            switches = [
                pokemon.species
                for pokemon in battle.available_switches[idx]
                if pokemon.species not in [
                    action.order.species
                    for action in next_action
                    if action is not None and not isinstance(action, DefaultBattleOrder) and isinstance(action.order, Pokemon)
                ]
            ]
            actions = [moves, switches]

            gimmick_output_format = ''
            if 'pokellmon' not in self.ps_client.account_configuration.username: # make sure we dont mess with pokellmon original strat
                dynamax_format = ' or {"dynamax":"<move_name>"}' if battle.can_dynamax else ''
                tera_format = ' or {"terastallize":"<move_name>"}' if battle.can_tera else ''
                gimmick_output_format = f'{dynamax_format}{tera_format}'

            # ADDITIONAL CHECK: Validate actions based on available options
            # Check if Pokemon is fainted or None first (highest priority)
            if battle.active_pokemon[idx] is None or battle.active_pokemon[idx].fainted:
                if len(switches) > 0:
                    #print(f"INFO: Pokemon fainted/None for slot {idx}, forcing switch selection only")
                    constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
                    constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step.Your thought should no more than 4 sentences. Your output MUST be a JSON like: {{"switch":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                    constraint_prompt_tot_1 = '''You MUST switch. Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                    constraint_prompt_tot_2 = '''You MUST switch. Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
                else:
                    #print(f"ERROR: Pokemon fainted/None but no switches available for slot {idx}, setting action to None")
                    next_action[idx] = None
                    continue
            # If no switches are available but moves are available
            elif len(switches) == 0 and len(moves) > 0:
                #print(f"INFO: No switches available for slot {idx}, forcing move selection only")
                constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>", "target":"<target_number>"}}{gimmick_output_format}
        Target numbers: 1=left opponent, 2=right opponent, 0=field effect, 0=self\n'''
                constraint_prompt_cot = f'''You MUST move. Choose the best action by thinking step by step.Your thought should no more than 4 sentences. Your output MUST be a JSON like: {{"move":"<move_name>", "target":"<target_number>"}}{gimmick_output_format}. Target numbers: 1=left opponent, 2=right opponent, 0=field effect, 0=self\n'''
                constraint_prompt_tot_1 = '''You MUST move. Generate top-k (k<=3) best move options. Your output MUST be a JSON like:{"option_1":{"action":"move","target":"<move_name>"}, ..., "option_k":{"action":"move","target":"<move_name>"}}. Available moves: {moves}\n'''
                constraint_prompt_tot_2 = '''You MUST move. Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"move","target":"<move_name>"}}\n'''
            # If no moves are available but switches are available
            elif len(moves) == 0 and len(switches) > 0:
                #print(f"INFO: No moves available for slot {idx}, forcing switch selection only")
                constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
                constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step.Your thought should no more than 4 sentences. Your output MUST be a JSON like: {{"switch":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                constraint_prompt_tot_1 = '''You MUST switch. Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}. Available switches: {switches}\n'''
                constraint_prompt_tot_2 = '''You MUST switch. Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            # If neither moves nor switches are available (error state)
            elif len(moves) == 0 and len(switches) == 0:
                #print(f"ERROR: No moves or switches available for slot {idx}, setting action to None")
                next_action[idx] = None
                continue
            # Normal case: both moves and switches are available
            else:
                constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>", "target":"<target_number>"}}{gimmick_output_format} or {{"switch":"<switch_pokemon_name>"}}
                Target numbers: 1=left opponent, 2=right opponent, 0=field effect, 0=self\n'''
                constraint_prompt_cot = f'''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {{"move":"<move_name>", "target":"<target_number>"}}{gimmick_output_format} or {{"switch":"<switch_pokemon_name>"}}. Target numbers: 1=left opponent, 2=right opponent, 0=field effect, 0=self. Available switches: {switches}\n'''
                constraint_prompt_tot_1 = '''You MUST move or switch. Generate top-k (k<=3) best move or switch options. Your output MUST be a JSON like:{"option_1":{"action":"move","target":"<move_name>"}, ..., "option_k":{"action":"move","target":"<move_name>"}} or {"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}. Available moves: {moves}. Available switches: {switches}\n'''
                constraint_prompt_tot_2 = '''You MUST move or switch. Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"move","target":"<move_name>"}} or {"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''

            state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io
            state_prompt_cot = state_prompt + state_action_prompt + constraint_prompt_cot
            state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_prompt_tot_1
            state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_prompt_tot_2
            #print(state_prompt_io)

            retries = 10
            # Chain-of-thought
            if self.prompt_algo == "io":
                next_action[idx] = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions, idx=idx)
            # print("next_action:", next_action[idx])
            if self.prompt_algo == "sc":
                next_action[idx] = self.sc(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions, idx=idx)
            if self.prompt_algo == "tot":
                next_action[idx] = self.tot(retries, system_prompt, state_prompt_tot_1, state_prompt_tot_2, battle, sim, actions=actions, idx=idx)
            
           
        next_action = DoubleBattleOrder(first_order=next_action[0], second_order=next_action[1])
        print(next_action)
        return next_action
   
    def io(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None, idx=0):
        next_action = None
        cot_prompt = 'In fewer than 3 sentences, let\'s think step by step:'
        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io + cot_prompt
        # print(state_prompt_io)
        # print('\n')
        # print('--------------------------------')
        # print('\n')
        for i in range(retries):
            try:
                llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_io,
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=300,
                                            # stop=["reason"],
                                            json_format=True,
                                            actions=actions,
                                            battle=battle)
        
                # load when llm does heavylifting for parsing
                if DEBUG:
                    print(f"Raw LLM output: {llm_output}")
                
                # Always show LLM reasoning in chat
                print(f"LLM [{self.ps_client.account_configuration.username}] Slot {idx+1}: {llm_output}")
                
                llm_action_json = json.loads(llm_output)
                if DEBUG:
                    print(f"Parsed JSON: {llm_action_json}")
                next_action = None

                dynamax = "dynamax" in llm_action_json.keys()
                tera = "terastallize" in llm_action_json.keys()
                is_a_move = dynamax or tera

                if "move" in llm_action_json.keys() or is_a_move:
                    if dynamax:
                        llm_move_id = llm_action_json["dynamax"].strip()
                    elif tera:
                        llm_move_id = llm_action_json["terastallize"].strip()
                    else:
                        llm_move_id = llm_action_json["move"].strip()
                    
                    # ENSURE TARGET IS PROVIDED: Check if target is specified
                    if "target" not in llm_action_json:
                        #print(f"WARNING: LLM did not provide target for move '{llm_move_id}', adding default target")
                        llm_action_json["target"] = 0  # Add default target
                    
                    # ADDITIONAL VALIDATION: Check if move is in available actions
                    if actions is not None and len(actions) > 0:
                        available_moves = actions[0]  # First element is moves list
                        # Convert to lowercase and remove spaces for case-insensitive comparison
                        llm_move_normalized = llm_move_id.lower().replace(' ', '')
                        available_moves_normalized = [move.lower().replace(' ', '') for move in available_moves]
                        if llm_move_normalized not in available_moves_normalized:
                            #print(f"WARNING: LLM requested move '{llm_move_id}' not in available moves {available_moves}")
                            continue  # Skip this iteration and try again
                    
                    move_list = battle.available_moves[idx]

                    # get target number with comprehensive parsing
                    llm_target = llm_action_json.get("target", None)
                    
                    # ENHANCED TARGET HANDLING: Parse various word targets into proper integers
                    if llm_target is None:
                        #print(f"WARNING: No target specified for move '{llm_move_id}', using default target")
                        llm_target = 0  # Default to EMPTY_TARGET_POSITION
                    elif isinstance(llm_target, str):
                        llm_target = self._parse_target_string(llm_target)
                    elif isinstance(llm_target, (int, float)):
                        llm_target = int(llm_target)
                    else:
                        #print(f"WARNING: Invalid target type '{type(llm_target)}' for move '{llm_move_id}', using default target")
                        llm_target = 0
                    
                    # Validate target is within valid range
                    if llm_target not in [-2, -1, 0, 1, 2]:
                        #print(f"WARNING: Target '{llm_target}' out of valid range [-2, -1, 0, 1, 2], using default target")
                        llm_target = 0
                    

                    if dont_verify: # opponent
                        move_list = battle.opponent_active_pokemon.moves.values()
                    
                    # Debug: print available moves
                    if DEBUG:
                        print(f"LLM requested move: '{llm_move_id}'")
                        print(f"LLM requested target: '{llm_target}'")
                        print(f"Available moves: {[move.id for move in move_list]}")
                    
                    for i, move in enumerate(move_list):
                        if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):                
                            next_action = self.create_order(move, dynamax=dynamax, terastallize=tera, move_target=llm_target)
                            if DEBUG:
                                print(f"Move match found: {move.id} with target: {llm_target}")
                            break
                    
                    if next_action is None and dont_verify:
                        # unseen move so just check if it is in the action prompt
                        if llm_move_id.lower().replace(' ', '') in state_action_prompt:
                            next_action = self.create_order(Move(llm_move_id.lower().replace(' ', ''), self.gen.gen), dynamax=dynamax, terastallize=tera)
                    
                    if next_action is None and DEBUG:
                        print(f"No move match found for '{llm_move_id}'")
                elif "switch" in llm_action_json.keys():
                    # Check if switches are available - if not, force move selection
                    if len(battle.available_switches[idx]) == 0:
                        #print(f"WARNING: LLM attempted to switch but no switches available for slot {idx}. Forcing move selection.")
                        # Skip switch processing and continue to next iteration to try move selection
                        continue
                    
                    llm_switch_species = llm_action_json["switch"].strip()
                    
                    # ADDITIONAL VALIDATION: Check if switch is in available actions
                    if actions is not None and len(actions) > 1:
                        available_switches = actions[1]  # Second element is switches list
                        # Convert to lowercase and remove spaces for case-insensitive comparison
                        llm_switch_normalized = llm_switch_species.lower().replace(' ', '')
                        available_switches_normalized = [switch.lower().replace(' ', '') for switch in available_switches]
                        if llm_switch_normalized not in available_switches_normalized:
                            #print(f"WARNING: LLM requested switch '{llm_switch_species}' not in available switches {available_switches}")
                            continue  # Skip this iteration and try again
                    
                    switch_list = battle.available_switches[idx]
                    if dont_verify: # opponent prediction
                        observable_switches = []
                        for _, opponent_pokemon in battle.opponent_team.items():
                            if not opponent_pokemon.active:
                                observable_switches.append(opponent_pokemon)
                        switch_list = observable_switches
                    
                    # Debug: print available switches
                    if DEBUG:
                        print(f"LLM requested switch: '{llm_switch_species}'")
                        print(f"Available switches: {[pokemon.species for pokemon in switch_list]}")
                    
                    for i, pokemon in enumerate(switch_list):
                        if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                            next_action = self.create_order(pokemon)
                            if DEBUG:
                                print(f"Switch match found: {pokemon.species}")
                            break
                    
                else:
                    raise ValueError('No valid action')
                
                # with open(f"{self.log_dir}/output.jsonl", "a") as f:
                #     f.write(json.dumps({"turn": battle.turn,
                #                         "system_prompt": system_prompt,
                #                         "user_prompt": state_prompt_io,
                #                         "llm_output": llm_output,
                #                         "battle_tag": battle.battle_tag
                #                         }) + "\n")
                
                if next_action is not None:
                    break
            except Exception as e:
                print(f'Exception: {e}', 'passed')
                continue
        if next_action is None:
            print('No action found. Choosing max damage move')
            try:
                print('No action found', llm_action_json, actions, dont_verify)
            except:
                pass
            print()
            # raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
            next_action = self.choose_max_damage_move(battle, idx=idx)
        return next_action

    def sc(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=None, idx=0):
        action_results = [self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions, idx=idx) for i in range(self.K)]
        action_message = [action.message for action in action_results]
        _, counts = np.unique(action_message, return_counts=True)
        index = np.argmax(counts)
        return action_results[index]
    
    def tot(self, retries, system_prompt, state_prompt_tot_1, state_prompt_tot_2, battle: Battle, sim, actions=None, idx=0):
        llm_output1 = ""
        next_action = None
        for i in range(retries):
            try:
                llm_output1 = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_tot_1,
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=200,
                                            json_format=True,
                                            battle=battle)
                break
            except:
                raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                continue    

        if llm_output1 == "":
            return self.choose_max_damage_move(battle)

        for i in range(retries):
            try:
                llm_output2 = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_tot_2.replace("[OPTIONS]", llm_output1),
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=100,
                                            json_format=True,
                                            battle=battle)

                next_action = self.parse_new(llm_output2, battle, sim)
                with open(f"{self.log_dir}/output.jsonl", "a") as f:
                    f.write(json.dumps({"turn": battle.turn,
                                        "system_prompt": system_prompt,
                                        "user_prompt1": state_prompt_tot_1,
                                        "user_prompt2": state_prompt_tot_2,
                                        "llm_output1": llm_output1,
                                        "llm_output2": llm_output2,
                                        "battle_tag": battle.battle_tag
                                        }) + "\n")
                if next_action is not None:     break
            except:
                raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                continue

        if next_action is None:
            next_action = self.choose_max_damage_move(battle)
        return next_action

    def estimate_matchup(self, sim: LocalVGCSim, battle: DoubleBattle, is_opp: bool=False) -> 'DoubleBattleOrder':
        """
        For each player's active Pokemon, select the move that is most effective against BOTH
        of the opponent's active Pokemon (i.e., the move with the lowest average turns to KO both).
        Then, combine the selected moves into a DoubleBattleOrder.
        """
        chosen_orders = [None, None]

        if is_opp:
            active_side = battle.opponent_active_pokemon
            available_moves = battle.opponent_available_moves
            opponent_side = battle.active_pokemon
        else:
            active_side = battle.active_pokemon
            available_moves = battle.available_moves
            opponent_side = battle.opponent_active_pokemon

        for idx in [0, 1]:
            if len(active_side) > idx and active_side[idx] is not None and not active_side[idx].fainted:
                mon = active_side[idx]
                moves = available_moves[idx] if len(available_moves) > idx else []
                best_move = None
                best_score = float('inf')
                for move in moves:
                    total_score = 0
                    count = 0
                    for opp_idx in [0, 1]:
                        if len(opponent_side) > opp_idx and opponent_side[opp_idx] is not None and not opponent_side[opp_idx].fainted:
                            opp_mon = opponent_side[opp_idx]
                            # Prefer moves that KO fastest on average
                            if getattr(move, "category", None) and move.category.name == "STATUS":
                                # Use the sim's status move estimator if available
                                turns = get_status_num_turns_fnt(mon, move, opp_mon, sim, boosts=mon._boosts.copy())
                            else:
                                turns = get_number_turns_faint(mon, move, opp_mon, sim, boosts1=mon._boosts.copy(), boosts2=opp_mon._boosts.copy())
                            total_score += turns
                            count += 1
                    if count > 0:
                        avg_score = total_score / count
                        if avg_score < best_score:
                            best_score = avg_score
                            best_move = self.create_order(move)
                if best_move is not None:
                    chosen_orders[idx] = best_move
        return self.create_double_order(chosen_orders)

       

    def dmg_calc_move(self, battle: DoubleBattle, return_move: bool=False):
        """
        Calculate the best damage-dealing moves for the current battle state.
        Uses estimate_matchup to select optimal moves for both active Pokemon.
        
        Args:
            battle: The DoubleBattle instance
            return_move: If True, return raw move orders instead of DoubleBattleOrder
        
        Returns:
            (DoubleBattleOrder, turns) where DoubleBattleOrder contains
                moves for both Pokemon (None if fainted)
            If return_move=True: ((move1_order, move2_order), turns)
        """
        sim = LocalVGCSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    format=self.format
        )
        
        # Use estimate_matchup to get the best moves for both active Pokemon
        double_order = self.estimate_matchup(sim, battle, is_opp=False)
        
        if return_move:
            if double_order.first_order is None and double_order.second_order is None:
                return None, np.inf
            move1_order = double_order.first_order.order if double_order.first_order is not None else None
            move2_order = double_order.second_order.order if double_order.second_order is not None else None
            return (move1_order, move2_order), 1
        
        # If no valid moves, return random move
        if double_order.first_order is None and double_order.second_order is None:
            random_move = self.choose_random_move(battle)
            return random_move, 1
        
        return double_order, 1
    
    
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

 
    def _get_fast_heuristic_evaluation(self, battle_state):
        """Fast heuristic evaluation for leaf nodes when LLM is not used."""
        try:
            player_hp = int(battle_state.active_pokemon.current_hp_fraction * 100) if battle_state.active_pokemon else 0
            opp_hp = int(battle_state.opponent_active_pokemon.current_hp_fraction * 100) if battle_state.opponent_active_pokemon else 0
            player_remaining = len([p for p in battle_state.team.values() if not p.fainted])
            opp_remaining = len([p for p in battle_state.opponent_team.values() if not p.fainted])
            
            # Use cached fast evaluation
            return fast_battle_evaluation(
                player_hp, opp_hp, 
                player_remaining, opp_remaining,
                battle_state.turn
            )
        except:
            # Ultimate fallback to basic hp difference
            try:
                from poke_env.player.local_simulation import LocalSim
                sim = LocalSim(battle_state, 
                            self.move_effect,
                            self.pokemon_move_dict,
                            self.ability_effect,
                            self.pokemon_ability_dict,
                            self.item_effect,
                            self.pokemon_item_dict,
                            self.gen,
                            self._dynamax_disable,
                            format=self.format
                )
                return sim.get_hp_diff()
            except:
                return 50  # Neutral fallback score
    
    def _initialize_minimax_optimizer(self, battle):
        """Initialize the minimax optimizer with current battle state."""
        try:
            initialize_minimax_optimization(
                battle=battle,
                move_effect=self.move_effect,
                pokemon_move_dict=self.pokemon_move_dict,
                ability_effect=self.ability_effect,
                pokemon_ability_dict=self.pokemon_ability_dict,
                item_effect=self.item_effect,
                pokemon_item_dict=self.pokemon_item_dict,
                gen=self.gen,
                _dynamax_disable=self._dynamax_disable,
                format=self.format,
                prompt_translate=self.prompt_translate
            )
            self._minimax_initialized = True
            print("[INIT] Minimax optimizer initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize minimax optimizer: {e}")
            self.use_optimized_minimax = False  # Fallback to original

    def check_timeout(self, start_time, battle):
        if time.time() - start_time > 30:
            print('default due to time')
            move, _ = self.dmg_calc_move(battle)
            return move
        else:
            return None
    
    def vgc_tree_search(self, retries, battle, sim=None, return_opp = False) -> DoubleBattleOrder:
        """
        Create a minimax tree of player/opponent action pairs
        Actions will be of the form DoubleBattleOrder
        Evalutates leaf nodes and determines best action based on score
        """
        start_time = time.time()
        root = VGCSimNode(battle=battle, 
                          move_effect=self.move_effect,
                          pokemon_move_dict=self.pokemon_move_dict,
                          ability_effect=self.ability_effect,
                          pokemon_ability_dict=self.pokemon_ability_dict,
                          item_effect=self.item_effect,
                          pokemon_item_dict=self.pokemon_item_dict,
                          gen=self.gen,
                          _dynamax_disable=self._dynamax_disable,
                          depth=1,
                          format=self.format,
                          prompt_translate=self.prompt_translate,
                          sim=sim
                          )
        # get battle state information for LLM decision
        system_prompt, state_prompt, _, _, _ = root.simulation.get_player_prompt()
        if DEBUG:
            print("system_prompt: ", system_prompt)
            print("state_prompt: ", state_prompt)
        # Check if any active pokemon is not fainted and has moves (for DoubleBattle, active_pokemon is a list)
        has_active_with_moves = False
        if isinstance(battle, DoubleBattle):
            has_active_with_moves = any(
                (mon is not None and not mon.fainted and len(battle.available_moves[i]) > 0)
                for i, mon in enumerate(battle.active_pokemon) if i < len(battle.available_moves)
            )
        else:
            has_active_with_moves = (battle.active_pokemon is not None and 
                                    not battle.active_pokemon.fainted and 
                                    len(battle.available_moves) > 0)
        if has_active_with_moves:
            # get dmg calc move for potential early return
            dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(battle)
            if dmg_calc_out is not None:
                try:
                    # Ask LLM to choose between damage calculator tool or minimax search upfront
                    tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                    1. Damage calculator advantages:
                    - Quick and efficient for finding optimal damaging moves
                    - Useful when a clear type advantage or high-power move is available
                    - Effective when the opponent is not switching and current pokemon is likely to KO opponent

                    2. Minimax tree search advantages:
                    - Can model opponent behavior and predict future moves
                    - Useful in complex situations with multiple viable options
                    - Effective when long-term strategy is crucial

                    3. Current battle state:
                    - Remaining Pokémon on each side
                    - Health of active Pokémon
                    - Type matchups
                    - Available moves and their effects
                    - Presence of status conditions or field effects

                    4. Uncertainty level:
                    - How predictable is the opponent's next move?
                    - Are there multiple equally viable options for your next move?

                    Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                    {"choice":"damage calculator"} or {"choice":"minimax"}'''

                    state_prompt_io = state_prompt + tool_prompt
                    llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                    user_prompt=state_prompt_io,
                                                    model=self.backend,
                                                    temperature=0.6,
                                                    max_tokens=100,
                                                    json_format=True,
                                                    battle=battle
                                                    )
                    # Load when llm does heavylifting for parsing
                    llm_action_json = json.loads(llm_output)
                    if 'choice' in llm_action_json.keys():
                        if llm_action_json['choice'] != 'minimax':
                            # LLM chose damage calculator - return it directly
                            print("LLM chose damage calculator over minimax")
                            if return_opp:
                                try:
                                    # For DoubleBattle, use first active pokemon for opponent estimation
                                    opp_mon = battle.opponent_active_pokemon[0] if isinstance(battle, DoubleBattle) and len(battle.opponent_active_pokemon) > 0 and battle.opponent_active_pokemon[0] is not None else battle.opponent_active_pokemon
                                    player_mon = battle.active_pokemon[0] if isinstance(battle, DoubleBattle) and len(battle.active_pokemon) > 0 and battle.active_pokemon[0] is not None else battle.active_pokemon
                                    if isinstance(battle, DoubleBattle):
                                        action_opp, _ = self.estimate_matchup(root.simulation, battle, opp_mon, player_mon, is_opp=True, pokemon_idx=0)
                                    else:
                                        action_opp, _ = self.estimate_matchup(root.simulation, battle, opp_mon, player_mon, is_opp=True)
                                    return dmg_calc_out, self.create_order(action_opp) if action_opp else None
                                except:
                                    return dmg_calc_out, None
                            return dmg_calc_out
                except Exception as e:
                    print(f'LLM choice failed ({e}), defaulting to minimax')
        print("Using minimax tree search")

        q = [root]
        leaf_nodes = []

        while len(q) != 0:
            node = q.pop(0)
            # get available actions efficiently
            player_actions = []
            system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)
            # check if terminal node or reached depth limit
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                    # Use LLM value function for leaf nodes evaluation
                    value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                    'Add points based on the effectiveness of current available moves.' +\
                                    'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                    'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                    'Subtract points for excessive switching.' +\
                                    'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                    'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                    cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                    state_prompt_io = state_prompt + value_prompt + cot_prompt
                    llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                    user_prompt=state_prompt_io,
                                                    model=self.backend,
                                                    temperature=self.temperature,
                                                    max_tokens=500,
                                                    json_format=True,
                                                    llm=self.llm_value,
                                                    battle=battle
                                                    )
                    # Load when llm does heavylifting for parsing
                    llm_action_json = json.loads(llm_output)
                    node.hp_diff = int(llm_action_json['score'])
                except Exception as e:
                    print("LLM value function failed, using hp diff")
                    #TODO implement HP diff scoring for leaf nodes
                leaf_nodes.append(node) 
                continue

            #--------------------------------------------------#
            # Generate actions for PLAYER using a combination  #
            # of damage calculator and LLM                     #
            #--------------------------------------------------#
            # Check if any active pokemon is not fainted and has moves (for DoubleBattle, active_pokemon is a list)
            has_active_with_moves = False
            if isinstance(node.simulation.battle, DoubleBattle):
                has_active_with_moves = any(
                    (mon is not None and not mon.fainted and i < len(node.simulation.battle.available_moves) and len(node.simulation.battle.available_moves[i]) > 0)
                    for i, mon in enumerate(node.simulation.battle.active_pokemon)
                )
            else:
                has_active_with_moves = (node.simulation.battle.active_pokemon is not None and 
                                        not node.simulation.battle.active_pokemon.fainted and 
                                        len(node.simulation.battle.available_moves) > 0)
            if has_active_with_moves:
                # Get dmg calc move
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                if dmg_calc_out is not None:
                    if DEBUG:
                        print("dmg_calc_out: ", dmg_calc_out)
                    player_actions.append(dmg_calc_out)
            
            # Generate LLM actions for both active pokemon in double battles
            try:
                action_orders = [None, None]  # [action for pokemon 0, action for pokemon 1]
                
                # Generate action for first active pokemon (idx=0)
                if (len(node.simulation.battle.active_pokemon) > 0 and 
                    node.simulation.battle.active_pokemon[0] is not None and 
                    not node.simulation.battle.active_pokemon[0].fainted):
                    try:
                        system_prompt_0, state_prompt_0, constraint_prompt_cot_0, constraint_prompt_io_0, state_action_prompt_0, _, _ = node.simulation.get_player_prompt(return_actions=True, idx=0)
                        action_0 = self.io(2, system_prompt_0, state_prompt_0, constraint_prompt_cot_0, constraint_prompt_io_0, state_action_prompt_0, node.simulation.battle, node.simulation, actions=None, idx=0)
                        if DEBUG:
                            print("llm action_0: ", action_0)
                        action_orders[0] = action_0
                    except Exception as e:
                        pass  # Skip if generation fails for first pokemon
                
                # Generate action for second active pokemon (idx=1)
                if (len(node.simulation.battle.active_pokemon) > 1 and 
                    node.simulation.battle.active_pokemon[1] is not None and 
                    not node.simulation.battle.active_pokemon[1].fainted):
                    try:
                        system_prompt_1, state_prompt_1, constraint_prompt_cot_1, constraint_prompt_io_1, state_action_prompt_1, _, _ = node.simulation.get_player_prompt(return_actions=True, idx=1)
                        action_1 = self.io(2, system_prompt_1, state_prompt_1, constraint_prompt_cot_1, constraint_prompt_io_1, state_action_prompt_1, node.simulation.battle, node.simulation, actions=None, idx=1)
                        action_orders[1] = action_1
                        if DEBUG:
                            print("llm action_1: ", action_1)
                    except Exception as e:
                        pass  # Skip if generation fails for second pokemon
                
                # Combine the two actions into a DoubleBattleOrder
                if action_orders[0] is not None or action_orders[1] is not None:
                    action_io_double = DoubleBattleOrder(first_order=action_orders[0], second_order=action_orders[1])
                    if DEBUG:
                            print("action_io_double: ", action_io_double)
                    # Check if this action is already in player_actions by comparing first_order and second_order
                    # (DoubleBattleOrder.__eq__ may not work correctly due to dataclass inheritance issues)
                    is_duplicate = False
                    for existing_action in player_actions:
                        if isinstance(existing_action, DoubleBattleOrder):
                            if (existing_action.first_order == action_io_double.first_order and 
                                existing_action.second_order == action_io_double.second_order):
                                is_duplicate = True
                                if DEBUG:
                                    print("action_io_double is duplicate of existing action, skipping")
                                break
                    if not is_duplicate:
                        print("adding to player_actions: ", action_io_double)
                        player_actions.append(action_io_double)
            except Exception as e:
                pass  # Use what we have

            #--------------------------------------------------#
            # Generate actions for OPPONENT using a combination#
            # of damage calculator and LLM                     #
            #--------------------------------------------------#
            opponent_actions = []
            try:
                # For DoubleBattle, use first active pokemon for opponent estimation
                if isinstance(node.simulation.battle, DoubleBattle):
                    opp_mon = node.simulation.battle.opponent_active_pokemon[0] if len(node.simulation.battle.opponent_active_pokemon) > 0 and node.simulation.battle.opponent_active_pokemon[0] is not None else None
                    player_mon = node.simulation.battle.active_pokemon[0] if len(node.simulation.battle.active_pokemon) > 0 and node.simulation.battle.active_pokemon[0] is not None else None
                    if opp_mon is not None and player_mon is not None:
                        action_opp, opp_turns = self.estimate_matchup(
                            node.simulation, node.simulation.battle, 
                            opp_mon, player_mon, 
                            is_opp=True, pokemon_idx=0
                        )
                    else:
                        action_opp = None
                        opp_turns = float('inf')
                else:
                    action_opp, opp_turns = self.estimate_matchup(
                        node.simulation, node.simulation.battle, 
                        node.simulation.battle.opponent_active_pokemon, 
                        node.simulation.battle.active_pokemon, 
                        is_opp=True
                    )
            except:
                action_opp = None
                opp_turns = float('inf')
            # Add action_opp if it's not None
            if action_opp is not None:
                if DEBUG:
                    print("action_opp: ", action_opp)
                opponent_actions.append(action_opp)
            try:
                system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(system_prompt)
                action_o = self.io(2, system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o, node.simulation.battle, node.simulation, dont_verify=True)
                if action_o is not None and action_o not in opponent_actions:
                    if DEBUG:
                        print("action_o: ", action_o)
                    opponent_actions.append(action_o)
            except:
                pass
            
            # create child nodes
            # Ensure both lists are not None and not empty
            if node.depth < self.K and player_actions and len(player_actions) > 0 and opponent_actions and len(opponent_actions) > 0:
                for action_p in player_actions[:2]:  # Limit to 2 player actions for performance
                    for action_o in opponent_actions[:2]:  # Limit to 2 opponent actions for performance
                        try:
                            child_node = node.create_child_node(action_p, action_o)
                            q.append(child_node)
                        except Exception as e:
                            print(f"Failed to create child node: {e}")
                            continue
            # Choose best action using original logic
            def get_tree_action(root_node):
                if len(root_node.children) == 0:
                    return root_node.action, root_node.hp_diff, root_node.action_opp
                    
                score_dict = {}
                action_dict = {}
                opp_dict = {}
                
                for child in root_node.children:
                    action = str(child.action.order)
                    if action not in score_dict:
                        score_dict[action] = []
                        action_dict[action] = child.action
                        opp_dict[action] = child.action_opp
                    score_dict[action].append(child.hp_diff)
                
                # Use max score for each action
                for action in score_dict:
                    score_dict[action] = max(score_dict[action])
                
                best_action_str = max(score_dict, key=score_dict.get)
                return action_dict[best_action_str], score_dict[best_action_str], opp_dict[best_action_str]
            
            action, _, action_opp = get_tree_action(root)   
            end_time = time.time()    

            if return_opp:
                return action, action_opp
            return action         
 
    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: DoubleBattle, idx: int):
        # pick max base power move, default to targeting opponent 1 position
        if battle.available_moves[idx]:
            best_move = max(battle.available_moves[idx], key=lambda move: move.base_power)
            return self.create_order(best_move, move_target=DoubleBattle.OPPONENT_1_POSITION)
        return self.choose_random_move(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        """Returns a teampreview order for the given battle using LLM analysis.
        
        This method queries the LLM to select the best 4 Pokemon and their order
        based on the available team and opponent's team information.
        
        :param battle: The battle.
        :type battle: AbstractBattle
        :return: The teampreview order in format /team XXXX
        :rtype: str
        """
        try:
            # Get available Pokemon from battle.available_switches, filtering out empty lists
            raw_available = list(battle.available_switches)
            available_pokemon = [pokemon for pokemon in raw_available if pokemon and not isinstance(pokemon, list)]
            if not available_pokemon:
                # Fallback to random selection if no valid Pokemon
                return self.random_teampreview(battle)
            
            # Get opponent team from battle._teampreview_opponent_team
            opponent_team = list(battle._teampreview_opponent_team)
            
            
            
            # Format Pokemon data for LLM
            team_data = self._format_team_data_for_llm(available_pokemon, opponent_team)
            
            # Create system prompt for team selection
            system_prompt = """You are an expert Pokemon VGC (Video Game Championships) team analyst. Your task is to select the best 4 Pokemon from the available team to bring to battle against the opponent's team.

        Key considerations for team selection:
        1. Type matchups and coverage
        2. Speed control and priority moves
        3. Synergy between Pokemon (weather, terrain, abilities)
        4. Countering opponent's threats
        5. Lead Pokemon strategy (who goes first)
        6. Backup options and flexibility

        Respond with ONLY a 4-digit number representing the indices of your selected Pokemon in order (e.g., "1234" means bring Pokemon 1, 2, 3, 4 in that order).

        The first two Pokemon will be your leads, the last two will be in the back.
        
        Do not repeat the same index more than once."""
            
            # Create user prompt with team data
            user_prompt = self._create_teampreview_user_prompt(team_data)
            
            # Query LLM for team selection
            llm_response = self.get_LLM_action(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.backend,
                temperature=0.7,
                max_tokens=200
            )
            
            # Parse LLM response and convert to team order
            team_order = self._parse_teampreview_response(llm_response, available_pokemon)
            
            if team_order:
                print(f"Team order: {team_order}")
                return f"/team {team_order}"
            else:
                # Fallback to random selection if parsing fails
                print("fallback to random teampreview")
                return self.random_teampreview(battle)
                
        except Exception as e:
            print(f"Error in teampreview: {e}")
            # Fallback to random selection on any error
            return self.random_teampreview(battle)

    def _format_team_data_for_llm(self, available_pokemon: List[Pokemon], opponent_team: List[Pokemon]) -> Dict[str, Any]:
        """Format Pokemon data for LLM analysis."""
        team_data = {
            "available_pokemon": [],
            "opponent_pokemon": []
        }
        
        # Format available Pokemon - numbering starts from 1 for first actual Pokemon
        for i, pokemon in enumerate(available_pokemon, 1):
            try:
                pokemon_info = {
                    "index": i,  # This will be 1, 2, 3, 4, 5, 6 for the actual Pokemon
                    "name": pokemon.species,
                    "type1": pokemon.type_1.name if pokemon.type_1 else "Unknown",
                    "type2": pokemon.type_2.name if pokemon.type_2 else None,
                    "ability": pokemon.ability or "Unknown",
                    "item": pokemon.item or "None",
                    "moves": [move.name for move in pokemon.moves.values()] if hasattr(pokemon, 'moves') and pokemon.moves else [],
                    "base_stats": {
                        "hp": pokemon.base_stats.get("hp", 0),
                        "atk": pokemon.base_stats.get("atk", 0),
                        "def": pokemon.base_stats.get("def", 0),
                        "spa": pokemon.base_stats.get("spa", 0),
                        "spd": pokemon.base_stats.get("spd", 0),
                        "spe": pokemon.base_stats.get("spe", 0)
                    } if hasattr(pokemon, 'base_stats') and pokemon.base_stats else {}
                }
                team_data["available_pokemon"].append(pokemon_info)
            except Exception as e:
                print(f"Debug: Error processing available pokemon {i} {pokemon}: {e}")
                import traceback
                traceback.print_exc()
                # Create a minimal pokemon info if there's an error
                pokemon_info = {
                    "index": i,
                    "name": str(pokemon) if hasattr(pokemon, '__str__') else "Unknown",
                    "type1": "Unknown",
                    "type2": None,
                    "ability": "Unknown",
                    "item": "None",
                    "moves": [],
                    "base_stats": {}
                }
                team_data["available_pokemon"].append(pokemon_info)
        
        # Format opponent Pokemon
        for pokemon in opponent_team:
            #print(f"Debug: Processing opponent pokemon: {pokemon}, type: {type(pokemon)}")
            try:
                pokemon_info = {
                    "name": pokemon.species,
                    "type1": pokemon.type_1.name if pokemon.type_1 else "Unknown",
                    "type2": pokemon.type_2.name if pokemon.type_2 else None,
                    "ability": pokemon.ability or "Unknown",
                    "item": pokemon.item or "None",
                    "moves": [move.name for move in pokemon.moves.values()] if hasattr(pokemon, 'moves') and pokemon.moves else [],
                    "base_stats": {
                        "hp": pokemon.base_stats.get("hp", 0),
                        "atk": pokemon.base_stats.get("atk", 0),
                        "def": pokemon.base_stats.get("def", 0),
                        "spa": pokemon.base_stats.get("spa", 0),
                        "spd": pokemon.base_stats.get("spd", 0),
                        "spe": pokemon.base_stats.get("spe", 0)
                    } if hasattr(pokemon, 'base_stats') and pokemon.base_stats else {}
                }
                team_data["opponent_pokemon"].append(pokemon_info)
            except Exception as e:
                print(f"Debug: Error processing opponent pokemon {pokemon}: {e}")
                # Create a minimal pokemon info if there's an error
                pokemon_info = {
                    "name": str(pokemon) if hasattr(pokemon, '__str__') else "Unknown",
                    "type1": "Unknown",
                    "type2": None,
                    "ability": "Unknown",
                    "item": "None",
                    "moves": [],
                    "base_stats": {}
                }
                team_data["opponent_pokemon"].append(pokemon_info)
        
        return team_data



    def _create_teampreview_user_prompt(self, team_data: Dict[str, Any]) -> str:
        """Create user prompt with team data for LLM analysis."""
        prompt = "Available Pokemon:\n"
        
        for pokemon in team_data["available_pokemon"]:
            prompt += f"{pokemon['index']}. {pokemon['name']} "
            prompt += f"({pokemon['type1']}"
            if pokemon['type2']:
                prompt += f"/{pokemon['type2']}"
            prompt += f") "
            prompt += f"Ability: {pokemon['ability']}, Item: {pokemon['item']}\n"
            if pokemon['moves']:
                prompt += f"   Moves: {', '.join(pokemon['moves'])}\n"
            if pokemon['base_stats']:
                stats = pokemon['base_stats']
                prompt += f"   Stats: HP:{stats.get('hp', 0)} Atk:{stats.get('atk', 0)} Def:{stats.get('def', 0)} "
                prompt += f"Spa:{stats.get('spa', 0)} Spd:{stats.get('spd', 0)} Spe:{stats.get('spe', 0)}\n"
            prompt += "\n"
        
        prompt += "\nOpponent's Team:\n"
        for pokemon in team_data["opponent_pokemon"]:
            prompt += f"- {pokemon['name']} "
            prompt += f"({pokemon['type1']}"
            if pokemon['type2']:
                prompt += f"/{pokemon['type2']}"
            prompt += f") "
            prompt += f"Ability: {pokemon['ability']}, Item: {pokemon['item']}\n"
            if pokemon['moves']:
                prompt += f"  Moves: {', '.join(pokemon['moves'])}\n"
            if pokemon['base_stats']:
                stats = pokemon['base_stats']
                prompt += f"  Stats: HP:{stats.get('hp', 0)} Atk:{stats.get('atk', 0)} Def:{stats.get('def', 0)} "
                prompt += f"Spa:{stats.get('spa', 0)} Spd:{stats.get('spd', 0)} Spe:{stats.get('spe', 0)}\n"
            prompt += "\n"
        
        prompt += "\nSelect your 4 Pokemon (respond with 4 digits):"
        
        return prompt

    def _parse_teampreview_response(self, response: str, available_pokemon: List[Pokemon]) -> Optional[str]:
        """Parse LLM response and convert to team order format."""
        try:
            # Clean the response
            response = response.strip()
            
            # Extract 4-digit number from response
            import re
            match = re.search(r'\b(\d{4})\b', response)
            if match:
                team_indices = match.group(1)
                
                # Validate indices
                valid_indices = []
                for idx_str in team_indices:
                    idx = int(idx_str)
                    if 1 <= idx <= len(available_pokemon):
                        valid_indices.append(idx_str)
                
                if len(valid_indices) == 4:
                    return ''.join(valid_indices)
            
            # If no valid 4-digit number found, try to extract individual numbers
            numbers = re.findall(r'\b(\d)\b', response)
            if len(numbers) >= 4:
                valid_indices = []
                for num in numbers[:4]:
                    idx = int(num)
                    if 1 <= idx <= len(available_pokemon):
                        valid_indices.append(num)
                
                if len(valid_indices) == 4:
                    return ''.join(valid_indices)
            
            return None
            
        except Exception as e:
            print(f"Error parsing teampreview response: {e}")
            return None

    