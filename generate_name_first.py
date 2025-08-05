import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os

# Mistral 7B 사용
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def visualize_solution_path(grid_str, grid_w, grid_h, solution_moves):
    """솔루션 경로를 그리드에 표시"""
    # 그리드를 2D 배열로 변환
    grid = list(grid_str)
    visit_count = [0] * len(grid)
    
    # 블록 위치 오프셋 (JavaScript의 pos_obj와 동일)
    pos_obj = {
        '1': [[0, 0]],           # 서있는 상태: 현재 위치만
        '2': [[0, 0], [1, 0]],   # 가로로 누운 상태: 현재 위치와 오른쪽
        '3': [[0, -1], [0, 0]]   # 세로로 누운 상태: 위쪽과 현재 위치
    }
    
    # 시작 위치도 표시
    start_pos = grid_str.find('1')
    if start_pos == -1:
        start_pos = grid_str.find('2')
    if start_pos == -1:
        start_pos = grid_str.find('3')
    
    if start_pos != -1:
        visit_count[start_pos] += 1
    
    # 각 이동 위치 표시
    for move in solution_moves:
        parts = move.split('#')
        if len(parts) >= 3:
            x, y = int(parts[0]), int(parts[1])
            block_type = parts[2].split('##')[0]
            
            # 블록이 차지하는 타일 표시
            if block_type in pos_obj:
                for offset in pos_obj[block_type]:
                    px = x + offset[0]
                    py = y + offset[1]
                    if 0 <= px < grid_w and 0 <= py < grid_h:
                        pos = py * grid_w + px
                        if 0 <= pos < len(grid):
                            visit_count[pos] += 1
    
    # 방문 횟수를 그리드에 표시
    solution_grid = []
    for i, char in enumerate(grid):
        if visit_count[i] > 0 and char in '._*HhABCDEabcde1234567890':
            if char == 'g':  # 목표 지점은 그대로 표시
                solution_grid.append('g')
            elif visit_count[i] < 10:
                solution_grid.append(str(visit_count[i]))
            else:
                solution_grid.append('+')
        else:
            solution_grid.append(char)
    
    return ''.join(solution_grid)

def generate_names_for_file(input_file, output_file, num_samples=100):
    """파일의 처음 num_samples개 레벨에만 이름 생성하고 그것만 저장"""
    
    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        all_levels = json.load(f)
    
    print(f"\n📄 Processing: {os.path.basename(input_file)}")
    print(f"   Total levels: {len(all_levels)}, Processing first {num_samples}")
    
    # 처음 100개만 선택
    levels_to_process = all_levels[:num_samples]
    
    # 이름 생성
    for i, level in enumerate(tqdm(levels_to_process, desc="Generating names")):
        # 1D 문자열을 2D 그리드로 변환
        grid_str = level['grid']
        grid_w = level['grid_w']
        grid_h = level['grid_h']
        grid_lines = []
        for row in range(0, len(grid_str), grid_w):
            grid_lines.append(grid_str[row:row+grid_w])
        grid_2d = '\n'.join(grid_lines)
        
        # 솔루션 경로 시각화
        solution_grid_str = visualize_solution_path(grid_str, grid_w, grid_h, level['move'])
        solution_lines = []
        for row in range(0, len(solution_grid_str), grid_w):
            solution_lines.append(solution_grid_str[row:row+grid_w])
        solution_2d = '\n'.join(solution_lines)
        
        # 프롬프트 수정 - 이름과 설명 모두 요청
        prompt = f"""<s>[INST] Bloxorz is a game where you roll a block from start to goal without falling off edges.

Level map:
{grid_2d}

Solution path (numbers show how many times each tile was visited):
{solution_2d}

Key: 1-9=visit count, g=goal, .=safe floor, _=void, *=glass (breaks if standing), H/h=bridge/switch

Based on the level layout and solution path, provide:
1. A creative 1-3 word name
2. A brief strategy tip or warning (1-2 sentences)

Format your response exactly like this:
Name: [creative name]
Tip: [strategy or warning]

Focus the tip on specific challenges visible in this level.

[/INST]"""
        
        # 생성
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # 더 길게 생성
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 응답 파싱
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Name과 Tip 추출
        name = ""
        tip = ""
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Name:"):
                name = line.replace("Name:", "").strip()
            elif line.startswith("Tip:"):
                tip = line.replace("Tip:", "").strip()
        
        # 백업 (파싱 실패 시)
        if not name:
            name = response.split('\n')[0].strip()
        if not tip:
            tip = "Navigate carefully to reach the goal."
        
        # 데이터에 추가
        level['name'] = name
        level['tip'] = tip
        
        # 10개마다 진행상황 출력
        if i % 10 == 0:
            print(f"\n{'='*60}")
            print(f"Level {i}:")
            print(f"Grid size: {level['grid_w']}x{level['grid_h']}")
            print(f"Solution length: {len(level['move'])} moves")
            print("\nFull input prompt:")
            print("-"*60)
            print(prompt)
            print("-"*60)
            print(f"\nGenerated response:")
            print(f"Name: {name}")
            print(f"Tip: {tip}")
            print('='*60)
    
    # 처리된 100개만 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(levels_to_process, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved {len(levels_to_process)} levels to: {output_file}")
    
    return levels_to_process

# 메인 실행
def process_all_files():
    """4개 파일을 각각 처리"""
    
    files = [
        ('data/bloxorz/puzzle_no_gimmick_all.json', 'data/bloxorz/puzzle_no_gimmick_named_100.json'),
        ('data/bloxorz/puzzle_glass_all.json', 'data/bloxorz/puzzle_glass_named_100.json'),
        ('data/bloxorz/puzzle_switch_all.json', 'data/bloxorz/puzzle_switch_named_100.json'),
        ('data/bloxorz/puzzle_bridge_all.json', 'data/bloxorz/puzzle_bridge_named_100.json')
    ]
    
    all_processed = []
    
    for input_file, output_file in files:
        if os.path.exists(input_file):
            processed = generate_names_for_file(input_file, output_file, num_samples=100)
            all_processed.extend(processed)
        else:
            print(f"⚠️  File not found: {input_file}")
    
    print(f"\n🎉 Total processed: {len(all_processed)} levels across all files")
    
    # 요약
    print("\n📝 Files created:")
    for _, output_file in files:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  - {os.path.basename(output_file)}: {len(data)} levels")

if __name__ == "__main__":
    process_all_files()