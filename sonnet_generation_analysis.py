'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer_sonnet_generation import AdamW, NorMuon, HTMuon, MuonWrapper

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

TRAIN_SIZE = 0.8

RUN_DIR = Path("runs/sonnet_generator")
RUN_DIR.mkdir(parents=True, exist_ok=True)

epoch_logs = []
generation_logs = []

training_start_time = time.perf_counter()
best_val_loss = float("inf")

TQDM_DISABLE = False


def log_epoch_metrics(epoch, train_loss, val_loss):
    epoch_logs.append({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/perplexity": np.exp(train_loss),
        "val/perplexity": np.exp(val_loss),
        "time_sec": time.perf_counter() - training_start_time,
    })


def log_generation(epoch, batch_idx, prompt, generated_text):
    generation_logs.append({
        "epoch": epoch,
        "batch_idx": batch_idx,
        "prompt": prompt,
        "generated_text": generated_text,
    })


def save_checkpoint(model, optimizer, args, path, epoch, val_loss):
    checkpoint = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "epoch": epoch,
        "val_loss": val_loss,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    torch.save(checkpoint, path)


def save_logs_and_plots(args):
    epoch_df = pd.DataFrame(epoch_logs)
    generation_df = pd.DataFrame(generation_logs)

    epoch_df.to_csv(RUN_DIR / "epoch_metrics.csv", index=False)
    generation_df.to_csv(RUN_DIR / "generated_samples.csv", index=False)

    with open(RUN_DIR / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if len(epoch_df) > 0:
        epoch_df.plot(x="epoch", y=["train/loss", "val/loss"])
        plt.title("Training and validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(RUN_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        epoch_df.plot(x="epoch", y=["train/perplexity", "val/perplexity"])
        plt.title("Training and validation perplexity")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.savefig(RUN_DIR / "perplexity_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

    print(f"Saved everything in: {RUN_DIR}")

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    last_hidden_state = self.gpt(input_ids, attention_mask)['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(last_hidden_state)
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device
  
  # Old generate function
  '''
  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output
  '''
  
  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_k=20, top_p=0.9, max_length=128):
    """
    Generates a sonnet using temperature scaling, top-k sampling,
    and top-p nucleus sampling.
    """

    self.eval()

    device = self.get_device()
    token_ids = encoding.to(device)

    attention_mask = torch.ones(
        token_ids.shape,
        dtype=torch.int64,
        device=device
    )

    for _ in range(max_length):
        logits_sequence = self.forward(token_ids, attention_mask)
        next_token_logits = logits_sequence[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, next_token_logits.size(-1))
            threshold = torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
            indices_to_remove = next_token_logits < threshold
            next_token_logits = next_token_logits.masked_fill(
                indices_to_remove,
                float("-inf")
            )

        # Top-p filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits,
                descending=True,
                dim=-1
            )

            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p

            # Keep the first token that exceeds top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1,
                index=sorted_indices,
                src=sorted_indices_to_remove
            )

            next_token_logits = next_token_logits.masked_fill(
                indices_to_remove,
                float("-inf")
            )

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == self.tokenizer.eos_token_id:
            break

        token_ids = torch.cat([token_ids, next_token], dim=1)

        new_attention = torch.ones(
            (token_ids.shape[0], 1),
            dtype=torch.int64,
            device=device
        )

        attention_mask = torch.cat([attention_mask, new_attention], dim=1)

    generated_output = self.tokenizer.decode(
    token_ids[0].cpu().numpy().tolist()
    )

    return token_ids, generated_output

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

# Training function and helper functions

def create_train_val_dataloaders(dataset, batch_size, collate_fn, seed=42):
    """
    Split the sonnet dataset into train, validation, and test sets.
    Split:
    - 70% training
    - 30% validation
    """

    n = len(dataset)
    train_size = int(TRAIN_SIZE * n)
    val_size = n - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def compute_loss(model, batch, device):
    """
    Compute next-token prediction loss for one batch.
    """

    b_ids, b_mask = batch['token_ids'], batch['attention_mask']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)

    # Ignore the last prediction
    logits = rearrange(
        logits[:, :-1].contiguous(),
        'b t d -> (b t) d'
    )

    # Ignore the first token in the labels
    labels = b_ids[:, 1:].contiguous().flatten()

    loss = F.cross_entropy(logits, labels, reduction='mean')

    return loss
  

def train_one_epoch(model, train_dataloader, optimizer, device, epoch):
    """
    Train the model for one epoch.
    """

    model.train()

    train_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        optimizer.zero_grad()

        loss = compute_loss(model, batch, device)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    return train_loss / num_batches
  
 
def evaluate_loss(model, dataloader, device):
    """
    Compute average loss without updating the model.
    Used for validation and test.
    """

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(model, batch, device)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches 
  
def train(args):
    """Train GPT-2 for sonnet generation with train/validation split."""

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    args = add_arguments(args)

    global RUN_DIR, epoch_logs, generation_logs, training_start_time

    RUN_DIR = Path(
    f"runs/sonnet_generator/"
    f"optimizer_{args.optimizer}_lr_{args.lr}_wd_{args.weight_decay}_seed_{args.seed}"
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    print(args.epochs)

    epoch_logs = []
    generation_logs = []
    training_start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Full dataset of complete sonnets
    sonnet_dataset = SonnetsDataset(args.sonnet_path)

    # Split complete sonnets into train / validation / test
    train_dataloader, val_dataloader = create_train_val_dataloaders(
        sonnet_dataset,
        args.batch_size,
        sonnet_dataset.collate_fn
    )

    # Held-out dataset: only first 3 lines, used only for qualitative generation
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    model = SonnetGPT(args)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    with open(RUN_DIR / "model_stats.json", "w") as f:
        json.dump({
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
        }, f, indent=2)

    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )

    elif args.optimizer == "normuon":
        optimizer = NorMuon(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )

    elif args.optimizer == "htmuon":
        optimizer = HTMuon(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimizer == "muon":
        optimizer = MuonWrapper(
            model, args
        )

    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0
    min_delta = args.min_delta
    
    best_epoch = None

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            epoch
        )

        val_loss = evaluate_loss(
            model,
            val_dataloader,
            device
        )

        print(
            f"Epoch {epoch}: "
            f"train loss :: {train_loss:.3f}, "
            f"val loss :: {val_loss:.3f}"
        )
        
        log_epoch_metrics(epoch, train_loss, val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            args=args,
                            path=RUN_DIR / "best_sonnet_gpt.pt",
                            epoch=epoch,
                            val_loss=val_loss
                        )
        else:
            patience_counter += 1

            print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # For training
        '''
        print('Generating several output sonnets...')

        model.eval()

        for i, batch in enumerate(held_out_sonnet_dataset):
            print(f"Batch {i}")
            print()
            encoding = model.tokenizer(
                batch[1],
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(device)

            output = model.generate(
                      encoding['input_ids'],
                      temperature=args.temperature,
                      top_k=args.top_k,
                      top_p=args.top_p
                  )

            print(f'{output[1]}\n\n')
            log_generation(
              epoch=epoch,
              batch_idx=i,
              prompt=batch[1],
              generated_text=output[1]
          )
        '''

    total_training_time = time.perf_counter() - training_start_time

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = None

    summary = {
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": float(np.exp(best_val_loss)),
        "total_training_time_sec": total_training_time,
        "peak_gpu_memory_mb": peak_memory_mb,
    }

    with open(RUN_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    save_logs_and_plots(args)
    save_model(model, optimizer, args, f'final_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  global RUN_DIR
  RUN_DIR = Path(
      f"runs/sonnet_generator/optimizer_adamw_lr_3e-05_wd_0.01_seed_11711"
  )

  saved = torch.load(RUN_DIR / "best_sonnet_gpt.pt", weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')
  Path(args.sonnet_out).parent.mkdir(parents=True, exist_ok=True)
  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, default=20,
                    help="Number of most likely tokens kept for top-k sampling.") # Added
  parser.add_argument("--patience", type=int, default=5,
                    help="Number of epochs to wait before early stopping.") # Added
  parser.add_argument("--min_delta", type=float, default=0.001,
                    help="Minimum validation loss improvement required.") # Added

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  
  parser.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw", "normuon", "htmuon", "muon"],
                    help="Optimizer to use.") # Added
  
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--beta1", type=float, default=0.9)
  parser.add_argument("--beta2", type=float, default=0.999)
  parser.add_argument("--eps", type=float, default=1e-6) # Added

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)