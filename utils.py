import ast
from collections.abc import Iterable
import hashlib
from itertools import groupby
import os
from typing import List
import numpy as np
import shutil
import yaml

import imageio
from peft import PeftModelForCausalLM, PeftConfig
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from conf.config import Config

from sokoban_solvers import EnhancedAStarAgent, State

# For naming runs, sort the hyperparameters in this order (e.g. model-gpt2_sample_prop-0.01_seed_4)
HYPERPARAM_SORT_ORDER = ['model', 'source', 'sample_prop', 'annotation_keys', 'seed', 'gen_temp', 'gen_top_p', 'gen_beams']

# Hyperparameters which, even when changed, should not be included in the run name
NON_NAMED_HYPERPARAMS = ["num_train_steps", "save_freq", "eval_freq", "no_log", "render", "num_eval_proc", "num_eval_samples",
                         "gen_freq", "gen_len", "gen_temp", "gen_beams", "gen_top_k", "gen_top_p", "gen_typical_p", "sample_contexts",
                         "sample_sequential", "eval_tolerance", "eval_controllability", "n_search_iters"]

def sort_hyperparams(hyperparams):
    """
    Sort the hyperparameters in a consistent order. 
    """
    return list(sorted(hyperparams, key=lambda k: HYPERPARAM_SORT_ORDER.index(k) if k in HYPERPARAM_SORT_ORDER 
                       else len(HYPERPARAM_SORT_ORDER)))

class CheckpointNotFoundError(FileNotFoundError):
    pass

def get_run_name(args: Config):

    # Case 1: we're training a model using a specific experiment config. In this case, we want to determine
    # which hyperparameters are swept over in the experiment a create a new log folder that references them
    # specifically.
    if args.exp_name != "":
        yaml_path = f"conf/experiment/{args.exp_name}.yaml"
        
        # yaml 파일이 있는 경우
        if os.path.exists(yaml_path):
            train_yaml = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
            train_sweep_params = sort_hyperparams(train_yaml['hydra']['sweeper']['params'].keys())
            sweep_values = [args[param] for param in train_sweep_params]
            run_name = os.path.join(
                args.exp_name,
                "_".join([f"{param}-{value}" for param, value in zip(train_sweep_params, sweep_values)])
            )
        # yaml 파일이 없는 경우 - exp_name을 폴더명으로 사용
        else:
            # Case 2와 동일한 로직 사용하되, "manual" 대신 exp_name 사용
            default_config = Config()
            changed_params = sort_hyperparams([param for param in args.keys() if args[param] != default_config.__dict__.get(param)])
            changed_params = [param for param in changed_params if param not in NON_NAMED_HYPERPARAMS]
            
            if changed_params == []:
                run_name = os.path.join(args.exp_name, "default")
            else:
                run_name = os.path.join(
                    args.exp_name,
                    "_".join([f"{param}-{args[param]}" for param in changed_params])
                )

    # Case 2: we're training a model using a specific set of hyperparameters. In this case, we want to
    # create a new log folder that references any hyperparameter that is different from the default
    else:
        default_config = Config()
        changed_params = sort_hyperparams([param for param in args.keys() if args[param] != default_config.__dict__.get(param)])
        changed_params = [param for param in changed_params if param not in NON_NAMED_HYPERPARAMS]
       
        if changed_params == []:
            run_name = os.path.join("manual", "default")

        else:
            run_name = os.path.join(
                "manual",
                "_".join([f"{param}-{args[param]}" for param in changed_params])
            )


    return run_name
    

def deprecated_get_run_name(args: Config):
    run_name = os.path.join(
        args.game,
        f"source:{args.source}" + (f"_char-encoded" if args.char_encoding else ""),
        f"model:{args.model}",
        f"level_key:{args.level_key}",
        f"annotation_keys:{args.annotation_keys}",
        f"num_annotation_buckets:{args.num_annotation_buckets}",
        f"holdouts:{args.holdout_solution_lens}",
        f"chunk_size-{args.chunk_size}_lr-{args.learning_rate}",
        f"sample_prop:{args.sample_prop}",
        f"seed-{args.seed}",
    )
    return run_name

def filter_configs(cfgs: List[Config]):
    new_cfgs = []
    for cfg in cfgs:
        if is_valid_config(cfg):
            cfg.run_name = get_run_name(cfg)
            new_cfgs.append(cfg)
    return new_cfgs

def is_valid_config(cfg: Config) -> bool:
    """ When manually sweeping over hyperparams, identify combinations."""
    if cfg.holdout_solution_lens is not None and cfg.annotation_keys is None:
        # Cannot hold out prompts when model is not trained to match prompts (?)
        return False
    return True

def process_hyperparam_str(hp_str: str) -> tuple:
    """
    Convert the value of a hyperparameter from a string to a list of values
    """
    try:
        values = ast.literal_eval(hp_str)
        if isinstance(values, Iterable):
            return values
        else:
            return (values,)

    except ValueError as err:
        exit(f"Invalid hyperparameter value: {hp_str} ({err})")


def save_train_state(model, optimizer, global_step, output_dir):
    """모델과 optimizer 상태를 저장"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 모델을 safetensors로 저장
    if hasattr(model, 'module'):
        model.module.save_pretrained(checkpoint_dir)
    else:
        model.save_pretrained(checkpoint_dir)
    
    # optimizer와 global_step은 별도 JSON으로 저장
    import json
    training_state = {
        'global_step': global_step,
        # optimizer state_dict는 너무 크므로 저장하지 않음
        # 재시작 시 learning rate scheduler만 맞춰주면 됨
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f)
    
    print(f"✓ Checkpoint saved at step {global_step}")


def load_train_state(output_dir, lora=False):
    """체크포인트에서 모델과 training state 로드"""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import json

    print(f"Looking for checkpoints in: {output_dir}")
    
    # 디렉토리 내용 확인
    if os.path.exists(output_dir):
        print(f"Directory contents: {os.listdir(output_dir)}")
    else:
        print(f"Directory does not exist: {output_dir}")
        raise CheckpointNotFoundError(f"Directory not found: {output_dir}")
    
    # 가장 최근 체크포인트 찾기
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoint_dirs.append((step, item))
                print(f"Found checkpoint: {item} (step {step})")
            except:
                continue
    
    if not checkpoint_dirs:
        raise CheckpointNotFoundError(f"No checkpoints found in {output_dir}")
    
    # 가장 큰 step의 체크포인트 선택
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_checkpoint = checkpoint_dirs[0]
    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
    
    print(f"Attempting to load checkpoint from {checkpoint_path}...")
    
    # 모델 로드 (safetensors 자동 사용)
    try:
        if lora:
            # LoRA 모델 로드
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(checkpoint_path)
            base_model_name = config._name_or_path
            
            # 베이스 모델 먼저 로드
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            # LoRA 어댑터 로드
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    except Exception as e:
        raise CheckpointNotFoundError(f"Could not load model from checkpoint: {e}")
    
    # training state 로드
    try:
        with open(os.path.join(checkpoint_path, "training_state.json"), "r") as f:
            training_state = json.load(f)
        global_step = training_state['global_step']
    except:
        global_step = latest_step  # 파일명에서 추출
    
    # optimizer state는 None 반환 (새로 초기화)
    return model, None, global_step

BOXOBAN_MAPPING = {
    '-': 'empty',
    '#': 'wall',
    '$': 'box',
    '.': 'goal',
    '*': 'box_in_place',
    '@': 'player',
    '+': 'player_in_place',
    ' ': 'empty'
}

BOXOBAN_INVERSE_MAPPING = {v: k for k, v in BOXOBAN_MAPPING.items()}

GRIDDLY_INVERSE_MAPPING = {
    'empty': '.',
    'wall': 'w',
    'box': 'b',
    'goal': 'h',
    'player': 'A',
    'box_in_place': 'f',
    'player_in_place': 'A'
}

BOXOBAN_TO_GRIDDLY_CHARS = {k: GRIDDLY_INVERSE_MAPPING[v] for k, v in BOXOBAN_MAPPING.items()}

GRIDDLY_ACTION_MAPPING = {(-1, 0): 1, (0, -1): 2, (1, 0): 3, (0, 1): 4}

def encode_boxoban_text(level):
    # Remove the first line of each level, which just contains the level number
    level = level[level.find("\n")+1:].strip()

    lines = []
    for line in level.split("\n"):
        # Group consecutive characters together and map them to their identities
        line_text = ", ".join([f"{len(list(iter))} {BOXOBAN_MAPPING[char]}" for char, iter in groupby(line)])
        lines.append(line_text)

    level_text = "\n".join(lines)

    return level_text

def decode_boxoban_text(text):
    # TODO: this code doesn't handle any error cases, which are sure to come up during generation
    level = ""

    for line in text.split("\n"):
        try:
            for section in line.split(", "):
                count, char = section.split(" ")
                level += BOXOBAN_INVERSE_MAPPING[char] * int(count)
        
            level += "\n"

        except:
            level += "Invalid line\n"

    return level

def generate_l_mazes(width, height):
    '''
    Generates the set of all "L Mazes" of a given width and height. We construct an L Maze by choosing a start
    and end point at least one square away from the edge and then connecting them with a path that has at most
    one turn. For example:

    ##########
    #      ###
    ###### ###
    ##########
    ##########
    '''

    def to_string(grid):
        return "\n".join(["".join(["#" if pos == 1 else "-" for pos in row]) for row in grid])

    def l_path(start, end):
        path = []

        cur_pos = start

        # Always gives the path that changes y before x
        while cur_pos[1] != end[1]:
            cur_pos = (cur_pos[0], cur_pos[1] + (1 if cur_pos[1] < end[1] else -1))
            path.append(cur_pos)

        while cur_pos[0] != end[0]:
            cur_pos = (cur_pos[0] + (1 if cur_pos[0] < end[0] else -1), cur_pos[1])
            path.append(cur_pos)

        return path


    l_mazes = []
    path_lens = []

    interior_positions = [(y, x) for x in range(1, width-1) for y in range(1, height-1)]
    used_starts = set()

    for start in interior_positions:
        for end in interior_positions:
            if start == end:
                continue
            if end in used_starts:
                continue
            used_starts.add(start)

            grid = np.ones((height, width), dtype=np.int8)
            path = l_path(start, end)

            grid[start] = 0
            for pos in path:
                grid[pos] = 0

            annotation = f"Width: {width}\nHeight: {height}\nPath length: {len(path)}\n"
            path_lens.append(len(path))
            l_mazes.append(annotation + to_string(grid))

    return l_mazes, path_lens

def _process_level(level):
    '''
    Given a boxoban level, return a dictionary containing all of the relevant information, 
    (see comment in __init__) for details
    '''

    solver = EnhancedAStarAgent()

    level_hash = _hash_level(level)

    level_text = encode_boxoban_text(level)
    level_state = State().stringInitialize(level.split("\n"))

    # Remove the first line of each level, which just contains the level number
    level = level[level.find("\n")+1:]

    # Pad the level with walls to make it rectangular
    max_width = max([len(row) for row in level.split("\n")])
    lines = []

    for line in level.split("\n"):
        if line == "": continue
        num_leading_spaces = len(line) - len(line.lstrip())
        formatted = ("#" * num_leading_spaces) + line.strip() + ("#" * (max_width - len(line)))
        lines.append(formatted)

    # Fill in gaps in to top and bottom rows
    lines[0] = lines[0].replace(" ", "#")
    lines[-1] = lines[-1].replace(" ", "#")

    # Combine the rows, strip, and replace spaces with dashes
    level = "\n".join(lines).strip().replace(" ", "-")

    width = len(level.split("\n")[0])
    height = len(level.split("\n"))
    num_targets = level.count("$") + level.count("*")
    prop_empty = level.count("-") / (width * height)

    solution, node, iterations = solver.getSolution(level_state, maxIterations=1_000_000, maxTime=-1)
    if node.checkWin():
        solution_len = len(solution)
        print(f"Solved after {iterations} iterations.")
    else:
        solution_len = -1
        solution = None
        print(f"Failed after {iterations} iterations.")

    return level, level_text, level_hash, width, height, num_targets, prop_empty, solution_len, solution

def _hash_level(level):
    return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)



def save_gif(env, lvl, sol, lvl_render_dir):
    if not os.path.isdir(lvl_render_dir):
        os.makedirs(lvl_render_dir)
    j = 0
    if sol != False:
        frames = []
        ep_rew = 0
        env.reset(level_string=lvl)
        im_name = os.path.join(lvl_render_dir, f"{j}.png")
        im = env.render(mode='rgb_array')
        im = Image.fromarray(im)
        im.save(im_name)
        frames.append(im)
        for act_id in sol:
            j += 1
            obs, rew, done, info = env.step(int(act_id))
            ep_rew += rew
            im_name = os.path.join(lvl_render_dir, f"{j}.png")
            im = env.render(mode='rgb_array')
            im = Image.fromarray(im)
            im.save(im_name)
            frames.append(im)
        
        # Parent of the level directory and name of the level directory
        render_dir, lvl_dir = os.path.split(lvl_render_dir)
        # Save gif with fps of 3
        imageio.mimsave(os.path.join(render_dir, f"{lvl_dir}.gif"), frames, fps=10)
