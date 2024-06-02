import sys
import os
import time
import torch
from tqdm import tqdm
from params import *
from model import *
from tiny_shakespeare_tokenizer import *
from dataclasses import asdict
import json

# Load the dataset
with open('/content/minLlama3/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

params = ModelArgs()
model = Llama3(params, tokenizer).to(params.device)

# Data loading for training
def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.max_seq_len, (batch_size,))
    x = torch.stack([data[i:i + params.max_seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + params.max_seq_len + 1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters=5):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Optimizer and learning rate scheduler
lr_init = 1e-2
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
max_iters = 2000
eval_interval = 100
warmup_iters = 50
warmup_factor = 1e-3
lr_final = 1e-5

def lr_lambda(current_iter):
    if current_iter < warmup_iters:
        return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
    else:
        decay_iters = max_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / decay_iters))
        return max(cosine_decay, lr_final / lr_init)
        
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop with throughput, latency, and GPU memory usage calculation
start_time = time.time()
total_samples = 0

for iter in tqdm(range(max_iters)):
    batch_start_time = time.time()
    
    # Sample a batch of data
    xb, yb = get_batch('train', params.max_batch_size)
    
    # Train
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    batch_end_time = time.time()
    batch_time = batch_end_time - batch_start_time
    total_samples += params.max_batch_size
    throughput = total_samples / (batch_end_time - start_time)
    
    # GPU memory usage
    gpu_memory_allocated = torch.cuda.memory_allocated(params.device) / (1024 ** 2)  # in MB
    gpu_memory_reserved = torch.cuda.memory_reserved(params.device) / (1024 ** 2)  # in MB
    
    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, params.max_batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"step {iter:04d}: lr {current_lr:.6f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, ppl {torch.exp(losses['val']).mean().item():.0f}, throughput {throughput:.2f} samples/sec, time elapsed: {elapsed_time:.2f} seconds, GPU memory allocated: {gpu_memory_allocated:.2f} MB, GPU memory reserved: {gpu_memory_reserved:.2f} MB")

# Inference latency measurement
def measure_inference_latency(model, prompt, num_samples=10):
    latencies = []
    for _ in range(num_samples):
        start_time = time.time()
        _ = model.generate(prompt)
        end_time = time.time()
        latencies.append(end_time - start_time)
    avg_latency = sum(latencies) / num_samples
    return avg_latency

prompt = "JULIET:\nO Romeo, Romeo! wherefore art thou R"
avg_latency = measure_inference_latency(model, prompt)
print(f"Average inference latency: {avg_latency:.4f} seconds")

# GPU memory usage during inference
gpu_memory_allocated = torch.cuda.memory_allocated(params.device) / (1024 ** 2)  # in MB
gpu_memory_reserved = torch.cuda.memory_reserved(params.device) / (1024 ** 2)  # in MB
print(f"Inference GPU memory allocated: {gpu_memory_allocated:.2f} MB, GPU memory reserved: {gpu_memory_reserved:.2f} MB")

name = f'models/{model.__class__.__name__}_{time.strftime("%Y-%m-%d|%H-%M-%S")}'
torch.save(model.state_dict(), f'{name}.pth')
params_dict = asdict(params)
with open(f'{name}.json', 'w') as f:
    json.dump(params_dict, f)
