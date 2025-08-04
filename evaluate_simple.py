# evaluate_simple.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from argparse import Namespace
from collections import defaultdict

from bloxorz_dataset import BloxorzDataset
from utils import load_train_state


def evaluate_model(model, tokenizer, dataset, device, args):
    """모델을 평가하고 통계를 반환"""
    model.eval()

    print(f"\n🎲 Generating {args.num_samples} samples...")
    samples = []

    with torch.no_grad():
        for i in tqdm(range(args.num_samples), desc="Generating"):
            # 랜덤 컨텍스트 생성
            context = dataset.gen_context()
            inputs = tokenizer(tokenizer.bos_token + context,
                               return_tensors="pt").input_ids
            inputs = inputs.to(device)

            # 레벨 생성
            outputs = model.generate(
                input_ids=inputs,
                max_length=args.gen_len,
                temperature=args.gen_temp,
                do_sample=True,
                top_p=args.gen_top_p,
                num_beams=args.gen_beams,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

            sample = dataset.decode(outputs)
            samples.append({
                'context': context,
                'generated': sample
            })

    print(f"\n📊 Evaluating samples...")
    results = []

    for i, sample_data in enumerate(tqdm(samples, desc="Evaluating")):
        sample = sample_data['generated']

        # 평가
        solution = dataset.get_solution(
            sample, n_search_iters=args.n_search_iters, verbose=False)
        accurate, info = dataset.is_accurate(sample, solution)
        novel, nearest_level, nearest_sol, distance = dataset.is_novel(sample)

        results.append({
            'context': sample_data['context'],
            'generated': sample,
            'novel': novel,
            'distance': distance,
            'valid': solution is not False,
            'solution_length': len(solution) if solution else -1,
            'accurate': accurate,
            'info': info
        })

    # 통계 계산
    num_novel = sum(r['novel'] for r in results)
    num_valid = sum(r['valid'] for r in results)
    num_accurate = sum(r['accurate'] for r in results)
    num_novel_valid = sum(r['novel'] and r['valid'] for r in results)
    num_all_three = sum(r['novel'] and r['valid']
                        and r['accurate'] for r in results)

    prop_novel = num_novel / len(results)
    prop_valid = num_valid / len(results)
    prop_accurate = num_accurate / len(results)
    prop_novel_valid = num_novel_valid / len(results)
    prop_all_three = num_all_three / len(results)

    # 다양성 계산
    unique_levels = set()
    for r in results:
        lines = r['generated'].strip().split('\n')
        level_start = 0
        for i, line in enumerate(lines):
            if ':' not in line:
                level_start = i
                break
        level = '\n'.join(lines[level_start:])
        unique_levels.add(level)

    diversity = len(unique_levels) / len(results)

    return {
        'results': results,
        'statistics': {
            'prop_novel': prop_novel,
            'prop_valid': prop_valid,
            'prop_accurate': prop_accurate,
            'prop_novel_valid': prop_novel_valid,
            'prop_all_three': prop_all_three,
            'diversity': diversity,
            'num_novel': num_novel,
            'num_valid': num_valid,
            'num_accurate': num_accurate,
            'num_novel_valid': num_novel_valid,
            'num_all_three': num_all_three,
            'num_unique': len(unique_levels),
            'total_samples': len(results)
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', help='Path to checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--exp_name', default=None,
                        help='Experiment name (for data selection)')
    parser.add_argument('--gen_temp', type=float,
                        default=1.0, help='Generation temperature')
    parser.add_argument('--gen_top_p', type=float,
                        default=0.9, help='Top-p sampling')
    parser.add_argument('--gen_beams', type=int,
                        default=1, help='Number of beams')
    parser.add_argument('--show_examples', type=int, default=5,
                        help='Number of examples to display')
    parser.add_argument('--save_results', default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # 체크포인트에서 설정 로드
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            train_config = json.load(f)
            model_name = train_config.get('model', 'gpt2')
            exp_name = args.exp_name or train_config.get(
                'exp_name', 'no_gimmick')
            lora = train_config.get('lora', False)
    else:
        print("Warning: config.json not found, using defaults")
        model_name = 'gpt2'
        exp_name = args.exp_name or 'no_gimmick'
        lora = False

    # 모델 이름 매핑
    model_mapping = {
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "gpt2-xl": "gpt2-xl",
        "opt-350m": "facebook/opt-350m",
        "opt-2.7b": "facebook/opt-2.7b",
    }

    # 데이터 파일 매핑
    data_map = {
        'no_gimmick': ['data/bloxorz/puzzle_no_gimmick_all.json'],
        'glass': ['data/bloxorz/puzzle_glass_all.json'],
        'switch': ['data/bloxorz/puzzle_switch_all.json'],
        'bridge': ['data/bloxorz/puzzle_bridge_all.json'],
        'all': [
            'data/bloxorz/puzzle_no_gimmick_all.json',
            'data/bloxorz/puzzle_glass_all.json',
            'data/bloxorz/puzzle_switch_all.json',
            'data/bloxorz/puzzle_bridge_all.json'
        ]
    }

    print(f"\n📁 Loading from: {args.checkpoint_dir}")
    print(f"  Model: {model_name}")
    print(f"  LoRA: {lora}")
    print(f"  Experiment: {exp_name}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_mapping[model_name])
    tokenizer.add_special_tokens({
        "pad_token": "PAD",
        "bos_token": "START",
        "eos_token": "END"
    })

    # Dataset
    print("📚 Loading dataset...")
    dataset = BloxorzDataset(
        tokenizer,
        model_name,
        data_file=data_map.get(exp_name, data_map['no_gimmick']),
        chunk_size=256
    )

    # Model
    print("🤖 Loading model...")

    # 체크포인트 찾기
    checkpoint_dirs = []
    for item in os.listdir(args.checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoint_dirs.append((step, item))
            except:
                continue

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")

    # 가장 최근 체크포인트
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_checkpoint = checkpoint_dirs[0]
    checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)

    print(f"  Loading from: {checkpoint_path}")

    # 모델 로드
    if lora:
        from peft import PeftModel
        # 베이스 모델 생성
        base_model = AutoModelForCausalLM.from_pretrained(model_mapping[model_name])
        base_model.resize_token_embeddings(len(tokenizer))  # 토크나이저 크기 맞추기
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    print(f"  Loaded from step: {latest_step}")

    # Evaluation args
    eval_args = Namespace(
        num_samples=args.num_samples,
        gen_len=300,
        gen_temp=args.gen_temp,
        gen_top_p=args.gen_top_p,
        gen_beams=args.gen_beams,
        n_search_iters=100000
    )

    # Evaluate
    results = evaluate_model(model, tokenizer, dataset, device, eval_args)

    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)

    stats = results['statistics']
    print(f"\n🎯 Overall Performance:")
    print(
        f"  Novel:    {stats['prop_novel']:.2%} ({stats['num_novel']}/{stats['total_samples']})")
    print(
        f"  Valid:    {stats['prop_valid']:.2%} ({stats['num_valid']}/{stats['total_samples']})")
    print(
        f"  Accurate: {stats['prop_accurate']:.2%} ({stats['num_accurate']}/{stats['total_samples']})")

    print(f"\n🏆 Combined Metrics:")
    print(
        f"  Novel + Valid:           {stats['prop_novel_valid']:.2%} ({stats['num_novel_valid']}/{stats['total_samples']})")
    print(
        f"  Novel + Valid + Accurate: {stats['prop_all_three']:.2%} ({stats['num_all_three']}/{stats['total_samples']})")

    print(f"\n🌈 Diversity:")
    print(
        f"  Unique levels: {stats['num_unique']}/{stats['total_samples']} ({stats['diversity']:.2%})")

    # Show examples
    if args.show_examples > 0:
        print(f"\n📝 Example Generations (showing {args.show_examples}):")
        print("-" * 60)

        for i, r in enumerate(results['results'][:args.show_examples]):
            print(f"\n[Example {i+1}]")
            print(
                f"Novel: {r['novel']} | Valid: {r['valid']} | Accurate: {r['accurate']}")
            print(f"Distance from nearest: {r['distance']}")
            if r['valid']:
                print(f"Solution length: {r['solution_length']}")
            print("\nGenerated level:")
            print(r['generated'])
            print("-" * 60)

    # Save results
    if args.save_results:
        save_path = args.save_results
        if not save_path.endswith('.json'):
            save_path += '.json'

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {save_path}")


if __name__ == "__main__":
    main()
