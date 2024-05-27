import sys
import os
current_dir = os.getcwd()  # Get the current working directory
venv_dir = os.path.join(current_dir, 'venv') 
python_version = str(sys.version_info.major) + '.' + str(sys.version_info.minor)
site_packages_path = os.path.join(venv_dir, 'lib', 'python' + python_version, 'site-packages')
sys.path.append(site_packages_path) 

import sys
import os
import torch
import time
import math
from tqdm import tqdm
from tiny_shakespeare_tokenizer import *
from params import *
from newmodel import *
from dataclasses import asdict
import json


# Load the dataset
with open('/content/minLlama3/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:200])

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

params = ModelArgs()
print(params)

model = Llama3(params, tokenizer).to(params.device)

# Print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e3, 'K parameters')
print(model)

# Data loading for training which generates a small batch of data of inputs x and targets y
def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.max_seq_len, (batch_size,))
    x = torch.stack([data[i:i+params.max_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+params.max_seq_len+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

@torch.no_grad()
def estimate_loss_and_perplexity(model, batch_size, eval_iters=5):
    model.eval()
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        for _ in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, targets=Y)
            losses[split].append(loss.item())
    avg_losses = {split: torch.tensor(losses[split]).mean().item() for split in losses}
    perplexities = {split: math.exp(avg_losses[split]) for split in avg_losses}
    model.train()
    return avg_losses, perplexities

# Create a PyTorch optimizer
lr_init = 1e-2
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)

# Training configuration
max_iters = 500
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

start_time = time.time()

for iter in range(max_iters):
    xb, yb = get_batch('train', params.max_batch_size)
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        avg_losses, perplexities = estimate_loss_and_perplexity(model, params.max_batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"step {iter:04d}: lr {current_lr:.6f}, train loss {avg_losses['train']:.4f}, val loss {avg_losses['val']:.4f}, "
              f"train perplexity {perplexities['train']:.4f}, val perplexity {perplexities['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")

# Save the model and parameters
name = f'/content/minLlama3/models/{model.__class__.__name__}_{time.strftime("%Y-%m-%d|%H-%M-%S")}'
torch.save(model.state_dict(), f'{name}.pth')

params_dict = asdict(params)
with open(f'{name}.json', 'w') as f:
    json.dump(params_dict, f)

print(model.generate("JULIET:\nO Romeo, Romeo! wherefore art thou R",max_gen_len=None))

# Perplexity calculation on the validation set
encodings = tokenizer.encode(text[val_data:], return_tensors='pt')
max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
device = params.device

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(f'Perplexity: {ppl.item()}')
