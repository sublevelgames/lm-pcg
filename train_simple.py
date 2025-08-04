# train_simple.py
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
   AutoTokenizer,
   AutoModelForCausalLM,
   set_seed,
   get_linear_schedule_with_warmup,
   DataCollatorForLanguageModeling
)
from tqdm import tqdm
import json
from argparse import Namespace

from bloxorz_dataset import BloxorzDataset
from evaluate import evaluate
from utils import save_train_state, load_train_state


def train_loop(model, tokenizer, optimizer, data_loader, output_dir, global_step, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # TensorBoard writer
    log_writer = SummaryWriter(output_dir, flush_secs=100) if output_dir else None
    
    # Calculate current epoch and batch
    epoch, batch_i = global_step // len(data_loader), global_step % len(data_loader)
    
    # Data loader iterator
    data_loader_iter = iter(data_loader)
    dataset = data_loader.dataset
    for _ in range(batch_i):
        next(data_loader_iter)
    
    # Learning rate scheduler
    num_train_steps = args.num_train_steps
    
    # optimizer가 새로 생성된 경우를 처리
    if global_step > 0:
        # 체크포인트에서 재개하는 경우
        # optimizer의 initial_lr을 수동으로 설정
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),  # 10% warmup
        num_training_steps=num_train_steps,
        last_epoch=global_step - 1 if global_step > 0 else -1  # -1로 수정
    )
    
    # 재개하는 경우 scheduler를 현재 step까지 진행
    if global_step > 0:
        for _ in range(global_step):
            scheduler.step()
    
    model.train()
    progress_bar = tqdm(total=num_train_steps, desc=f"Training {args.model}")
    progress_bar.update(global_step)
    
    done_training = False
    
    try:
        while not done_training:
            epoch += 1
            for batch_i in range(batch_i, len(data_loader)):
                global_step += 1
                
                batch = next(data_loader_iter)
                token_ids = batch["input_ids"].to(device)
                labels = token_ids.clone().detach()
                labels[labels == tokenizer.pad_token_id] = -100
                
                loss = model(token_ids, labels=labels)[0]
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at step {global_step}, skipping")
                    continue
                
                del token_ids
                del labels
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                if log_writer:
                    log_writer.add_scalar("train/loss", loss, global_step)
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Generate sample every N steps
                if global_step % args.gen_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        context = dataset.gen_context()
                        inputs = tokenizer(tokenizer.bos_token + context, return_tensors="pt").input_ids
                        inputs = inputs.to(device)
                        
                        outputs = model.generate(
                            input_ids=inputs,
                            max_length=args.gen_len,
                            temperature=args.gen_temp,
                            do_sample=True,
                            top_p=args.gen_top_p,
                            num_beams=1,
                            pad_token_id=tokenizer.eos_token_id,
                        )[0]
                        
                        sample = dataset.decode(outputs)
                        print(f"\n🎯 GENERATED SAMPLE AT STEP {global_step}:")
                        print(sample)
                        
                        # 평가 추가!
                        solution = dataset.get_solution(sample, n_search_iters=args.n_search_iters, verbose=False)
                        accurate, info = dataset.is_accurate(sample, solution)
                        novel, nearest_level, nearest_sol, distance = dataset.is_novel(sample)
                        
                        print(f"\n📊 VALIDATION RESULTS:")
                        print(f"  Novel: {novel} (distance: {distance})")
                        print(f"  Valid: {solution is not False}")
                        print(f"  Accurate: {accurate}")
                        if info:
                            for key, value in info.items():
                                print(f"  {key}: {value}")
                        print("-" * 50)
                        
                        if log_writer:
                            log_writer.add_text("eval/random_sample", f"```\n{sample}\n```", global_step)
                    
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_freq == 0 and output_dir:
                    save_train_state(model, optimizer, global_step, output_dir)
                
                # # Full evaluation
                # if global_step % args.eval_freq == 0:
                #     print(f"\n📈 Full evaluation at step {global_step}...")
                    
                #     # evaluate 함수 직접 구현
                #     model.eval()
                #     samples = []
                #     with torch.no_grad():
                #         for _ in range(args.num_eval_samples):
                #             context = dataset.gen_context()
                #             inputs = tokenizer(tokenizer.bos_token + context, return_tensors="pt").input_ids.to(device)
                            
                #             output = model.generate(
                #                 input_ids=inputs,
                #                 max_length=args.gen_len,
                #                 temperature=args.gen_temp,
                #                 do_sample=True,
                #                 top_p=args.gen_top_p,
                #                 num_beams=1,
                #                 pad_token_id=tokenizer.eos_token_id,
                #             )[0]
                            
                #             samples.append(dataset.decode(output))
                    
                #     # 평가
                #     solutions = [dataset.get_solution(s, n_search_iters=args.n_search_iters) for s in samples]
                #     novelties = [dataset.is_novel(s)[0] for s in samples]
                #     accuracies = [dataset.is_accurate(s, sol)[0] for s, sol in zip(samples, solutions)]
                    
                #     prop_accurate = sum(accuracies) / len(samples)
                #     prop_playable = sum([sol is not False for sol in solutions]) / len(samples)
                #     prop_novel = sum(novelties) / len(samples)
                #     diversity = dataset.get_diversity(samples) / len(samples)
                    
                #     print(f"Proportion accurate: {prop_accurate:.2%}")
                #     print(f"Proportion playable: {prop_playable:.2%}")
                #     print(f"Proportion novel: {prop_novel:.2%}")
                #     print(f"Diversity: {diversity:.2f}")
                    
                #     if log_writer:
                #         log_writer.add_scalar("eval/prop_playable", prop_playable, global_step)
                #         log_writer.add_scalar("eval/prop_novel", prop_novel, global_step)
                #         log_writer.add_scalar("eval/prop_accurate", prop_accurate, global_step)
                #         log_writer.add_scalar("eval/diversity", diversity, global_step)
                    
                #     model.train()
                
                if global_step >= args.num_train_steps:
                    done_training = True
                    break
            
            data_loader_iter = iter(data_loader)
            batch_i = 0
            
    except KeyboardInterrupt:
        print("\nStopping early...")
    
    progress_bar.close()
    if output_dir:
        save_train_state(model, optimizer, global_step, output_dir)
    print(f"Finished training: {global_step} steps")


def main():
   import argparse
   
   parser = argparse.ArgumentParser()
   # Hydra 스타일의 key=value 인자들
   parser.add_argument('overrides', nargs='*', help='Override config values')
   
   cmd_args = parser.parse_args()
   
   # 기본 설정
   args = Namespace(
       # Model
       model='gpt2',  # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
       lora=False,    # LoRA 사용 여부
       
       # Data
       game='bloxorz',
       exp_name='no_gimmick',  # 'no_gimmick', 'bridge', 'switch', 'glass', 'all'
       data_files=None,  # exp_name에 따라 자동 설정
       
       # Training
       batch_size=1,
       learning_rate=5e-5,
       weight_decay=0.01,
       num_train_steps=10000,
       chunk_size=256,
       
       # Generation
       gen_freq=1000,
       gen_len=300,
       gen_temp=1.0,
       gen_top_p=0.9,
       
       # Evaluation
       save_freq=5000,
       eval_freq=5000,
       num_eval_samples=100,
       n_search_iters=100000,
       
       # Misc
       seed=42,
       overwrite=False,
       resume=True,
       
       # Eval config compatibility
       gen_top_k=50,
       gen_typical_p=1.0,
       gen_beams=1,
       sample_contexts=False,
       sample_sequential=False,
       annotation_keys=None,
       eval_tolerance=None,
       num_eval_proc=1
   )
   
   # Override 파싱 (key=value 형식)
   for override in cmd_args.overrides:
       if '=' in override:
           key, value = override.split('=', 1)
           # 타입 변환
           if hasattr(args, key):
               current_value = getattr(args, key)
               if isinstance(current_value, bool):
                   setattr(args, key, value.lower() == 'true')
               elif isinstance(current_value, int):
                   setattr(args, key, int(value))
               elif isinstance(current_value, float):
                   setattr(args, key, float(value))
               else:
                   setattr(args, key, value)
           else:
               print(f"Warning: Unknown parameter {key}")
   
   # exp_name에 따라 데이터 파일 자동 선택
   if args.data_files is None:
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
       
       if args.exp_name not in data_map:
           print(f"Error: Unknown exp_name '{args.exp_name}'")
           print(f"Available options: {', '.join(data_map.keys())}")
           return
           
       args.data_files = data_map[args.exp_name]
   
   print(f"\nConfiguration:")
   print(f"  Model: {args.model}")
   print(f"  LoRA: {args.lora}")
   print(f"  Experiment: {args.exp_name}")
   print(f"  Data files: {args.data_files}")
   print(f"  Steps: {args.num_train_steps}")
   print(f"  Batch size: {args.batch_size}")
   print("-" * 50)
   
   set_seed(args.seed)
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   
   # 모델 이름 매핑
   model_mapping = {
       "gpt2": "gpt2",
       "gpt2-medium": "gpt2-medium",
       "gpt2-large": "gpt2-large", 
       "gpt2-xl": "gpt2-xl"
   }
   model_name = model_mapping[args.model]
   
   # Tokenizer
   print(f"Loading tokenizer for {model_name}...")
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   tokenizer.add_special_tokens({
       "pad_token": "PAD",
       "bos_token": "START",
       "eos_token": "END"
   })
   
   # Dataset
   print("Loading dataset...")
   dataset = BloxorzDataset(
       tokenizer,
       args.model,
       data_file=args.data_files,
       chunk_size=args.chunk_size
   )
   
   # Output directory
   output_dir = f"./logs/{args.exp_name}_{args.model}"
   if args.lora:
       output_dir += "_lora"
   
   # Model 및 체크포인트 처리
   print(f"Loading model {model_name}...")
   global_step = 0
   
   # 체크포인트 확인
   checkpoint_exists = False
   if args.resume and os.path.exists(output_dir) and not args.overwrite:
       checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
       if checkpoint_dirs:
           checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
           latest_checkpoint = checkpoint_dirs[0]
           checkpoint_path = os.path.join(output_dir, latest_checkpoint)
           checkpoint_exists = True
           
           try:
               print(f"Attempting to resume from {checkpoint_path}...")
               
               if args.lora:
                   # LoRA 모델 로드
                    from peft import PeftModel
                    # 베이스 모델 생성
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    model.resize_token_embeddings(len(tokenizer))
                    # LoRA 어댑터 로드 - is_trainable=True 추가!
                    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
                    model.train()  # 학습 모드로 설정
               else:
                   # 일반 모델 로드
                   model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
               
               # global_step 복원
               training_state_path = os.path.join(checkpoint_path, "training_state.json")
               if os.path.exists(training_state_path):
                   with open(training_state_path, "r") as f:
                       training_state = json.load(f)
                       global_step = training_state['global_step']
               else:
                   global_step = int(latest_checkpoint.split("-")[1])
               
               print(f"Resumed from step {global_step}")
               
           except Exception as e:
               print(f"Failed to load checkpoint: {e}")
               if input("Start from scratch? (y/n): ").lower() != 'y':
                   return
               checkpoint_exists = False
               global_step = 0
   
   # 체크포인트가 없거나 로드 실패한 경우 새 모델 생성
   if not checkpoint_exists:
       model = AutoModelForCausalLM.from_pretrained(model_name)
       model.resize_token_embeddings(len(tokenizer))
       
       # Apply LoRA if requested
       if args.lora:
           from peft import LoraConfig, get_peft_model, TaskType
           
           print("Applying LoRA...")
           peft_config = LoraConfig(
               task_type=TaskType.CAUSAL_LM, 
               inference_mode=False, 
               r=8, 
               lora_alpha=32, 
               lora_dropout=0.1
           )
           model = get_peft_model(model, peft_config)
           model.print_trainable_parameters()
   
   # Data loader
   data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
   data_loader = DataLoader(
       dataset, 
       collate_fn=data_collator, 
       batch_size=args.batch_size, 
       shuffle=True, 
       num_workers=0  # Windows에서는 0 권장
   )
   
   # Optimizer
   optimizer = torch.optim.AdamW(
       model.parameters(), 
       lr=args.learning_rate, 
       weight_decay=args.weight_decay
   )
   
   # Create output directory
   os.makedirs(output_dir, exist_ok=True)
   
   # Save config
   with open(os.path.join(output_dir, "config.json"), "w") as f:
       json.dump(vars(args), f, indent=2)
   
   # Start training
   print("\nStarting training...")
   print(f"Model: {args.model}")
   print(f"Data: {args.data_files}")
   print(f"Batch size: {args.batch_size}")
   print(f"Learning rate: {args.learning_rate}")
   print(f"Total steps: {args.num_train_steps}")
   print(f"Starting from step: {global_step}")
   print("-" * 50)
   
   train_loop(model, tokenizer, optimizer, data_loader, output_dir, global_step, args)

if __name__ == "__main__":
   main()