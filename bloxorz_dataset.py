import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import hashlib

class BloxorzDataset(Dataset):
    def __init__(self, 
                 tokenizer,
                 model_name: str,
                 data_file="data/bloxorz/puzzles.json",  # 단일 파일 또는 리스트
                 split="train",
                 chunk_size=256,
                 cache_dir="./caches",
                 cfg=None):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.pad_token_id = tokenizer.pad_token_id
        self.level_hashes = set()
        
        # Solver 초기화
        self.solver = BloxorzSolver()
        
        # 어노테이션 키들
        self.annotation_keys = ["grid_size", "block_types", "glass_tiles", 
                               "switches", "bridges", "collectables", 
                               "move_length", "difficulty"]
        self.novelty_threshold = 10
        
        # 데이터 로드 - 단일 파일 또는 여러 파일 지원
        self.raw_data = []
        
        # data_file이 문자열인지 리스트인지 확인
        if isinstance(data_file, str):
            data_files = [data_file]
        else:
            data_files = data_file
            
        # 모든 파일에서 데이터 로드
        for file_path in data_files:
            if not os.path.exists(file_path):
                print(f"Warning: Data file not found: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    self.raw_data.extend(file_data)
                else:
                    self.raw_data.append(file_data)
                print(f"Loaded {len(file_data) if isinstance(file_data, list) else 1} levels from {file_path}")
        
        if not self.raw_data:
            raise FileNotFoundError(f"No valid data files found in: {data_files}")
        
        print(f"Total levels loaded: {len(self.raw_data)}")
        
        # 레벨 해시 생성
        for item in self.raw_data:
            level_hash = self._hash_level(item['grid'])
            self.level_hashes.add(level_hash)
        
        # 토큰화 캐시 경로 - 파일 리스트를 반영
        files_hash = hashlib.md5('_'.join(sorted(data_files)).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"bloxorz_{model_name}_{files_hash}_tokens.npy")
        
        if os.path.exists(cache_path):
            print(f"Loading tokens from cache: {cache_path}")
            self.all_token_ids = np.load(cache_path)
        else:
            print("Tokenizing Bloxorz levels...")
            self._tokenize_data(cache_path)

        # 데이터셋 통계 수집
        self._collect_statistics()

    def _collect_statistics(self):
        """데이터셋의 실제 통계 수집"""
        self.stats = {
            'grid_sizes': [],
            'block_types': [],
            'glass_tiles': [],
            'switches': [],
            'bridges': [],
            'collectables': [],
            'move_lengths': [],
            'difficulties': []
        }
        
        for item in self.raw_data:
            # Grid size
            self.stats['grid_sizes'].append((item['grid_w'], item['grid_h']))
            
            # Block types
            block_types = set()
            for move in item['move']:
                parts = move.split('#')
                if len(parts) >= 3:
                    block_types.add(parts[2].split('##')[0])
            self.stats['block_types'].append(len(block_types))
            
            # Gimmicks
            if 'gimmick' in item:
                glass = item.get('glass_count', 0) if 'glass' in item['gimmick'] else 0
                switches = len(item['switch'].split('|')) if 'switch' in item['gimmick'] and 'switch' in item else 0
                bridges = len(item['bridge'].split('|')) if 'bridge' in item['gimmick'] and 'bridge' in item else 0
            else:
                glass = switches = bridges = 0
                
            self.stats['glass_tiles'].append(glass)
            self.stats['switches'].append(switches)
            self.stats['bridges'].append(bridges)
            
            # Collectables
            collectables = len(item['collectables'].split('#')) if 'collectables' in item and item['collectables'] else 0
            self.stats['collectables'].append(collectables)
            
            # Move length
            move_length = len(item['move'])
            self.stats['move_lengths'].append(move_length)
            
            # Difficulty
            if move_length < 10:
                self.stats['difficulties'].append('easy')
            elif move_length < 20:
                self.stats['difficulties'].append('medium')
            else:
                self.stats['difficulties'].append('hard')
    
    def _tokenize_data(self, cache_path):
        all_token_ids = []
        
        for item in tqdm(self.raw_data, desc="Tokenizing"):
            # 어노테이션 생성
            annotation = self._format_annotation(item)
            
            # 2D 그리드로 변환
            level_string = self._format_level(item['grid'], item['grid_w'], item['grid_h'])
            
            # 전체 문자열 구성
            full_text = f"{self.tokenizer.bos_token}{annotation}{level_string}{self.tokenizer.eos_token}"
            
            # 토큰화
            token_ids = self.tokenizer.encode(
                full_text, 
                padding="max_length", 
                max_length=self.chunk_size, 
                truncation=True
            )
            
            all_token_ids.extend(token_ids)
        
        self.all_token_ids = np.array(all_token_ids, dtype=np.int32)
        
        # 캐시 저장
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, self.all_token_ids)
    
    def _format_level(self, grid_str, grid_w, grid_h):
        """1D 문자열을 2D 그리드로 변환"""
        level_lines = []
        for i in range(0, len(grid_str), grid_w):
            level_lines.append(grid_str[i:i+grid_w])
        return '\n'.join(level_lines)
    
    def _format_annotation(self, item):
        annotation = f"Grid size: {item['grid_w']}x{item['grid_h']}\n"
        
        # 블록 타입 수
        block_types = set()
        for move in item['move']:
            parts = move.split('#')
            if len(parts) >= 3:
                block_types.add(parts[2].split('##')[0])
        annotation += f"Block types: {len(block_types)}\n"
        
        # 기믹 정보 추가
        has_gimmicks = False
        
        if 'gimmick' in item and item['gimmick']:
            has_gimmicks = True
            
            # Glass tiles
            if 'glass' in item['gimmick']:
                annotation += f"Glass tiles: {item.get('glass_count', 0)}\n"
            else:
                annotation += f"Glass tiles: 0\n"
                
            # Switches
            if 'switch' in item['gimmick'] and 'switch' in item:
                switch_count = len(item['switch'].split('|'))
                annotation += f"Switches: {switch_count}\n"
            else:
                annotation += f"Switches: 0\n"
                
            # Bridges
            if 'bridge' in item['gimmick'] and 'bridge' in item:
                bridge_count = len(item['bridge'].split('|'))
                annotation += f"Bridges: {bridge_count}\n"
            else:
                annotation += f"Bridges: 0\n"
        else:
            # 기믹이 없는 경우
            annotation += f"Glass tiles: 0\n"
            annotation += f"Switches: 0\n"
            annotation += f"Bridges: 0\n"
        
        # 수집품
        if 'collectables' in item and item['collectables']:
            collectables = item['collectables'].split('#')
            annotation += f"Collectables: {len(collectables)}\n"
        else:
            annotation += f"Collectables: 0\n"
        
        # 난이도
        move_length = len(item['move'])
        annotation += f"Move length: {move_length}\n"
        
        if move_length < 10:
            difficulty = "easy"
        elif move_length < 20:
            difficulty = "medium"
        else:
            difficulty = "hard"
        annotation += f"Difficulty: {difficulty}\n"
        
        return annotation
    
    def gen_context(self):
        """데이터셋의 실제 분포를 반영한 랜덤 컨텍스트 생성"""
        import random
        
        # 실제 데이터에서 랜덤하게 선택
        size = random.choice(self.stats['grid_sizes'])
        block_types = random.choice(self.stats['block_types'])
        glass_tiles = random.choice(self.stats['glass_tiles'])
        switches = random.choice(self.stats['switches'])
        bridges = random.choice(self.stats['bridges'])
        collectables = random.choice(self.stats['collectables'])
        move_length = random.choice(self.stats['move_lengths'])
        
        # Difficulty는 move_length에 따라 결정
        if move_length < 10:
            difficulty = "easy"
        elif move_length < 20:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        context = f"Grid size: {size[0]}x{size[1]}\n"
        context += f"Block types: {block_types}\n"
        context += f"Glass tiles: {glass_tiles}\n"
        context += f"Switches: {switches}\n"
        context += f"Bridges: {bridges}\n"
        context += f"Collectables: {collectables}\n"
        context += f"Move length: {move_length}\n"
        context += f"Difficulty: {difficulty}\n"
        
        return context
    
    def get_solution(self, level, n_search_iters=100000, verbose=False):
        """플레이 가능성 검증 - 실제 solver 사용"""
        # 어노테이션 제거
        lines = level.strip().split('\n')
        level_start = 0
        for i, line in enumerate(lines):
            if ':' not in line:
                level_start = i
                break
        
        level_lines = lines[level_start:]
        
        # 기본 유효성 검사
        if not self._is_valid_level(level, verbose):
            return False
        
        # 실제 solver로 해결 시도
        is_solvable, move_list = self.solver.solve(level_lines, n_search_iters)
        
        if verbose:
            if is_solvable:
                print(f"++레벨 해결 가능: {len(move_list)}개 이동++")
            else:
                print("--레벨 해결 불가능--")
        
        return move_list if is_solvable else False
    
    def _is_valid_level(self, level_text, verbose=False):
        # 어노테이션 제거
        lines = level_text.strip().split('\n')
        level_start = 0
        for i, line in enumerate(lines):
            if ':' not in line:
                level_start = i
                break
        
        level_lines = lines[level_start:]
        
        # 1. 직사각형 검사
        if not level_lines:
            if verbose: print("--빈 레벨--")
            return False
            
        line_lengths = [len(line) for line in level_lines]
        if len(set(line_lengths)) != 1:
            if verbose: print("--직사각형이 아님--")
            return False
        
        # 2. 유효한 문자 검사
        valid_chars = set('123._*gABCDEabcdeHh')
        level_str = ''.join(level_lines)
        if not set(level_str).issubset(valid_chars):
            if verbose: print("--유효하지 않은 문자--")
            return False
        
        # 3. 시작점과 목표점 검사
        start_count = level_str.count('1') + level_str.count('2') + level_str.count('3')
        if start_count != 1:
            if verbose: print("--시작점이 정확히 1개가 아님--")
            return False
            
        if level_str.count('g') != 1:
            if verbose: print("--목표점이 정확히 1개가 아님--")
            return False
        
        if verbose: print("++유효한 레벨++")
        return True
    
    def is_accurate(self, annotated_level, solution, tolerance=None):
        """어노테이션 정확성 검사"""
        lines = annotated_level.strip().split('\n')
        
        # 어노테이션 파싱
        expected = {}
        level_start = 0
        for i, line in enumerate(lines):
            if ':' in line:
                key, value = line.split(': ', 1)
                key_normalized = key.lower().replace(' ', '_')
                
                # Grid size 특별 처리
                if key_normalized == 'grid_size':
                    w, h = value.split('x')
                    expected['grid_w'] = int(w)
                    expected['grid_h'] = int(h)
                elif key_normalized == 'difficulty':
                    expected[key_normalized] = value
                else:
                    expected[key_normalized] = int(value)
            else:
                level_start = i
                break
        
        # 실제 값 계산
        level_lines = lines[level_start:]
        level_str = ''.join(level_lines)
        
        actual_h = len(level_lines)
        actual_w = len(level_lines[0]) if level_lines else 0
        
        # 실제 통계
        glass_count = level_str.count('*')
        
        # 비교
        accurate = True
        if 'grid_w' in expected and expected['grid_w'] != actual_w:
            accurate = False
        if 'grid_h' in expected and expected['grid_h'] != actual_h:
            accurate = False
        if 'glass_tiles' in expected and expected['glass_tiles'] != glass_count:
            accurate = False
            
        level_info = {
            "grid_w": actual_w,
            "grid_h": actual_h,
            "glass_tiles": glass_count,
            "valid": solution if isinstance(solution, bool) else True
        }
        
        return accurate, level_info
    
    def is_novel(self, annotated_level):
        """novelty 검사 - Levenshtein distance 기반"""
        from Levenshtein import distance
        
        lines = annotated_level.strip().split('\n')
        level_start = 0
        for i, line in enumerate(lines):
            if ':' not in line:
                level_start = i
                break
        
        level = '\n'.join(lines[level_start:])
        level_1d = level.replace('\n', '')
        
        level_hash = self._hash_level(level_1d)
        is_novel_by_hash = level_hash not in self.level_hashes
        
        # 실제로 가장 가까운 레벨 찾기
        nearest_level = level_1d
        nearest_solution = []
        min_distance = float('inf')
        
        if hasattr(self, 'raw_data') and self.raw_data:
            for item in self.raw_data:
                dist = distance(level_1d, item['grid'])
                if dist < min_distance:
                    min_distance = dist
                    nearest_level = item['grid']
                    nearest_solution = item.get('move', [])
        
        # novelty threshold 체크
        is_novel = min_distance >= self.novelty_threshold
        
        # 거리 정보도 반환
        return is_novel, nearest_level, nearest_solution, min_distance
    
    def _hash_level(self, level):
        """레벨 해시 생성"""
        return int(hashlib.md5(level.encode("utf-8")).hexdigest(), 16)
    
    def decode(self, token_ids):
        """토큰 디코딩"""
        eos_pos = torch.where(token_ids == self.tokenizer.eos_token_id)
        
        if len(eos_pos[0]) == 0:
            eos_pos = len(token_ids)
        else:
            eos_pos = eos_pos[0][0].item()
        
        token_ids = token_ids[:eos_pos]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.strip()
    
    def get_diversity(self, levels, clique_limit=1000000):
        """다양성 계산"""
        if len(levels) <= 1:
            return 1
        
        unique_levels = set()
        for level in levels:
            lines = level.strip().split('\n')
            level_start = 0
            for i, line in enumerate(lines):
                if ':' not in line:
                    level_start = i
                    break
            
            level_2d = '\n'.join(lines[level_start:])
            unique_levels.add(level_2d)
        
        return len(unique_levels)
    
    def __len__(self):
        return len(self.all_token_ids) // self.chunk_size
    
    def __getitem__(self, idx):
        start, end = self.chunk_size * idx, self.chunk_size * (idx+1)
        return torch.tensor(self.all_token_ids[start:end], dtype=torch.long)

class BloxorzSolver:
    def __init__(self):
        self.nn_obj = {
            '1': [[-2, 0], [0, -1], [1, 0], [0, 2]],    # W N E S
            '2': [[-1, 0], [0, -1], [2, 0], [0, 1]],    # W N E S
            '3': [[-1, 0], [0, -2], [1, 0], [0, 1]]     # W N E S
        }
        
        self.move_obj = {
            '1': [
                [[-1,0],[-2,0], 2],  # W
                [[0,-1],[0,-2], 3],  # N
                [[1,0],[2,0], 2],    # E
                [[0,1],[0,2], 3]     # S
            ],
            '2': [
                [[-1,0], 1],         # W
                [[0,-1], [1,-1], 2], # N
                [[2,0], 1],          # E
                [[0,1], [1,1], 2]    # S
            ],
            '3': [
                [[-1,0], [-1,-1], 3], # W
                [[0,-2], 1],          # N
                [[1,0], [1,-1], 3],   # E
                [[0,1], 1]            # S
            ]
        }
        
        self.switch_wall_type = 'ABCDE'
        self.switch_type = 'abcde'
        self.bridge_type = 'H'
        self.bridge_switch_type = 'h'
    
    def solve(self, grid_lines, max_iters=100000):
        """
        그리드를 받아서 해결 가능한지 확인하고 solution을 반환
        Returns: (is_solvable, move_list) tuple
        """
        # 2D 그리드를 1D 배열로 변환
        grid_h = len(grid_lines)
        grid_w = len(grid_lines[0]) if grid_lines else 0
        grid = []
        for line in grid_lines:
            grid.extend(list(line))
        
        # 시작점과 목표점 찾기
        start_pos = None
        goal_pos = None
        for i, cell in enumerate(grid):
            if cell in '123':
                start_pos = i
                start_type = cell
            elif cell == 'g':
                goal_pos = i
        
        if start_pos is None or goal_pos is None:
            return False, []
        
        # 스위치와 브릿지 정보 파싱
        switch_positions = {}
        bridge_positions = {}
        
        for i, cell in enumerate(grid):
            if cell in self.switch_type:
                # 해당 스위치가 제어하는 벽 찾기
                for j, wall in enumerate(grid):
                    if wall == cell.upper():
                        switch_positions[cell] = (i, j)
            elif cell == self.bridge_switch_type:
                # 브릿지 타일 찾기
                bridges = []
                for j, tile in enumerate(grid):
                    if tile == self.bridge_type:
                        bridges.append(j)
                if bridges:
                    bridge_positions[i] = bridges
        
        # BFS로 해결
        x = start_pos % grid_w
        y = start_pos // grid_w
        
        # arr: [(position_string, move_history, switch_history, bridge_state)]
        arr = [[f"{x}#{y}#{start_type}", [], [], {}]]
        visited = {}
        visited[arr[0][0]] = 1
        
        solved = False
        winning_move = []
        iterations = 0
        
        while arr and not solved and iterations < max_iters:
            iterations += 1
            arr2 = []
            
            for state in arr:
                x, y, block_type = map(int, state[0].split('#'))
                current_switches = state[2][:]
                current_bridge_state = state[3].copy()
                
                # 4방향 이동 시도
                for direction in range(4):
                    # 목표 위치 계산
                    x1 = x + self.nn_obj[str(block_type)][direction][0]
                    y1 = y + self.nn_obj[str(block_type)][direction][1]
                    idx1 = y1 * grid_w + x1
                    
                    # 경계 체크
                    if x1 < 0 or x1 >= grid_w or y1 < 0 or y1 >= grid_h:
                        continue
                    
                    # 이동 정보
                    move_arr = self.move_obj[str(block_type)][direction]
                    moved_type = str(move_arr[-1])
                    move_possible = True
                    
                    new_switches = current_switches[:]
                    new_bridge_state = current_bridge_state.copy()
                    
                    # 경로상의 모든 타일 체크
                    for k in range(len(move_arr) - 1):
                        x2 = x + move_arr[k][0]
                        y2 = y + move_arr[k][1]
                        idx2 = y2 * grid_w + x2
                        
                        # 경계 체크
                        if x2 < 0 or x2 >= grid_w or y2 < 0 or y2 >= grid_h:
                            move_possible = False
                            break
                        
                        # 빈 공간 체크
                        if grid[idx2] == '_':
                            move_possible = False
                            break
                        
                        # 유리 타일 체크 (블록 타입 1만 통과 불가)
                        if grid[idx2] == '*' and moved_type == '1':
                            move_possible = False
                            break
                        
                        # 스위치 벽 체크
                        if grid[idx2] in self.switch_wall_type:
                            switch_char = grid[idx2].lower()
                            if switch_char not in new_switches:
                                move_possible = False
                                break
                        
                        # 브릿지 타일 체크
                        if grid[idx2] == self.bridge_type:
                            # 이 브릿지를 제어하는 스위치 찾기
                            bridge_active = False
                            for switch_idx, bridges in bridge_positions.items():
                                if idx2 in bridges:
                                    if new_bridge_state.get(switch_idx, False):
                                        bridge_active = True
                                    break
                            
                            if not bridge_active:
                                move_possible = False
                                break
                    
                    if not move_possible:
                        continue
                    
                    # 목표 도달 체크
                    if x1 == goal_pos % grid_w and y1 == goal_pos // grid_w and moved_type == '1':
                        solved = True
                        winning_move = state[1] + [state[0]]
                        break
                    
                    # 스위치 활성화 체크
                    if moved_type == '1':
                        if grid[idx1] in self.switch_type and grid[idx1] not in new_switches:
                            new_switches.append(grid[idx1])
                        elif grid[idx1] == self.bridge_switch_type:
                            new_bridge_state[idx1] = not new_bridge_state.get(idx1, False)
                    
                    # 상태 문자열 생성
                    state_str = f"{x1}#{y1}#{moved_type}#{'|'.join(new_switches)}#{str(new_bridge_state)}"
                    
                    if state_str in visited:
                        continue
                    
                    visited[state_str] = 1
                    arr2.append([f"{x1}#{y1}#{moved_type}", 
                               state[1] + [state[0]], 
                               new_switches, 
                               new_bridge_state])
            
            arr = arr2
        
        if solved:
            return True, winning_move
        else:
            return False, []