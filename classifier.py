#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier on SST and CFIMDB
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
#from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from models.gpt2 import GPT2Model
from optimizer import AdamW, NorMuon
from tqdm import tqdm


import torch.nn as nn

# Edit
import mlflow
import time
import math
from typing import Any, Dict, List, Optional
import gc


# Make code and UI use the same backend
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_experiment("Deep Learning Project")

# IMPORTANT: Enable system metrics monitoring
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)




TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class GPT2SentimentClassifier(torch.nn.Module):
  '''
  This module performs sentiment classification using GPT2 in a cloze-style (fill-in-the-blank) task.

  In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
  Thus, your forward() should return one logit for each of the 5 classes.
  '''

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # Pretrain mode does not require updating GPT paramters.
    # Finetune last_layer or complete model. Set by config.fine_tune_mode
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE
    #raise NotImplementedError
    
    # Insert a new layer at the end. 
    # It takes in the encodings of the last token and outputs 5 gaussians or softmax.
    # Use softmax activation
    # Regularization?
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)


  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of the last token.
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    ### YOUR CODE HERE
    #raise NotImplementedError

    encoder_output = self.gpt(input_ids, attention_mask)
    last_token = encoder_output["last_token"]
    logits = self.classifier(last_token)

    return logits



class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    sent_ids = [x[1] for x in data]

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  if flag == 'test':
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device, return_dict=False, flag="val"):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []

  with torch.no_grad():
    total_loss = 0
    num_batches = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
      b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                    batch['labels'], batch['sents'], batch['sent_ids']

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      batch_size = b_ids.shape[0]

      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size

      logits = logits.detach().cpu().numpy()
      preds = np.argmax(logits, axis=1).flatten()

      b_labels = b_labels.cpu().flatten()
      y_true.extend(b_labels)
      y_pred.extend(preds)
      sents.extend(b_sents)
      sent_ids.extend(b_sent_ids)

      total_loss += loss
      num_batches += 1.0

  f1_macro = f1_score(y_true, y_pred, average='macro')
  f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
  precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
  recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
  conf_matrix = confusion_matrix(y_true, y_pred)
  accuracy = accuracy_score(y_true, y_pred)

  total_loss = total_loss.cpu() / num_batches

  if return_dict:
    return {
        f"{flag}/loss": total_loss,
        f"{flag}/accuracy": accuracy,
        f"{flag}/f1_macro": f1_macro,
        f"{flag}/f1_weighted": f1_weighted,
        f"{flag}/precision_macro": precision_macro,
        f"{flag}/recall_macro": recall_macro,
        #f"{flag}/confusion_matrix": conf_matrix,
    }
  else:
    return accuracy, f1_macro, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def get_trainable_model_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def clone_parameters(parameters):
    return [
        param.detach().clone()
        for param in parameters
    ]

def calculate_global_grad_norm(parameters):
    total_norm_squared = 0.0

    for param in parameters:
        if param.grad is not None:
            grad_norm = param.grad.detach().norm(2)
            total_norm_squared += grad_norm.item() ** 2

    return total_norm_squared ** 0.5

def calculate_update_metrics(params_before, params_after):
    update_norm_squared = 0.0
    weight_norm_squared = 0.0
    grad_dot_update = 0.0
    grad_norm_squared = 0.0

    for param_before, param_after in zip(params_before, params_after):
        update = param_after.detach() - param_before.detach()

        update_norm_squared += update.norm(2).item() ** 2
        weight_norm_squared += param_before.norm(2).item() ** 2

        if param_after.grad is not None:
            grad = param_after.grad.detach()

            grad_dot_update += torch.sum(grad * update).item()
            grad_norm_squared += grad.norm(2).item() ** 2

    update_norm_global = update_norm_squared ** 0.5
    weight_norm_global = weight_norm_squared ** 0.5
    grad_norm_global = grad_norm_squared ** 0.5

    update_to_weight_ratio = update_norm_global / (weight_norm_global + 1e-12)

    grad_update_cosine = grad_dot_update / (
        (grad_norm_global * update_norm_global) + 1e-12
    )

    return {
        "update_norm_global": update_norm_global,
        "update_to_weight_ratio": update_to_weight_ratio,
        "grad_update_cosine": grad_update_cosine,
    }



class OptimWrapper:
  def __init__(self, model, args):
    self.muon_params = []
    self.adamw_decay_params = []
    self.adamw_no_decay_params = []

    self.muon_names = []
    self.adamw_decay_names = []
    self.adamw_no_decay_names = []
  
    self.adamw = None
    self.muon = None

    self.split_params(model)

    if args.optim == "AdamW":
      self.adamw = torch.optim.AdamW(
          [
              {
                  "params": self.adamw_decay_params,
                  "lr": args.lr,
                  "weight_decay": args.weight_decay,
              },
              {
                  "params": self.adamw_no_decay_params,
                  "lr": args.lr,
                  "weight_decay": 0.0,
              },
              {
                  "params": self.muon_params,
                  "lr": args.lr,
                  "weight_decay": args.weight_decay,
              },
          ],
          betas=(args.beta_1, args.beta_2),
          eps=args.eps,
      )
    elif args.optim == "Muon":
      self.muon = torch.optim.Muon(
          self.muon_params,
          lr=args.lr,
          momentum=args.momentum,
          weight_decay=args.weight_decay,
          ns_steps=args.ns_steps,
          eps=args.eps
      )
      
      self.adamw = torch.optim.AdamW(
          [
              {
                  "params": self.adamw_decay_params,
                  "lr": args.adam_lr,
                  "weight_decay": args.adam_weight_decay,
              },
              {
                  "params": self.adamw_no_decay_params,
                  "lr": args.adam_lr,
                  "weight_decay": 0.0,
              },
          ],
          betas=(args.adam_beta_1, args.adam_beta_2),
          eps=args.eps,
      )
    elif args.optim == "NorMuon":
      self.muon = NorMuon(    
          self.muon_params,
          lr=args.lr,
          betas = (args.beta_1, args.beta_2),
          eps=args.eps,
          weight_decay=args.weight_decay,
          ns_steps=args.ns_steps
      )

      self.adamw = torch.optim.AdamW(
          [
              {
                  "params": self.adamw_decay_params,
                  "lr": args.adam_lr,
                  "weight_decay": args.adam_weight_decay,
              },
              {
                  "params": self.adamw_no_decay_params,
                  "lr": args.adam_lr,
                  "weight_decay": 0.0,
              },
          ],
          betas=(args.adam_beta_1, args.adam_beta_2),
          eps=args.eps,
      )
    else:
       raise ValueError("Error: Incorrect Value For Optimizer Selection.")
    
  def step(self):
    if self.adamw is not None: 
        self.adamw.step()
    if self.muon is not None:
        self.muon.step()
  
  def zero_grad(self):
    if self.adamw is not None: 
        self.adamw.zero_grad(set_to_none=True)
    if self.muon is not None:
        self.muon.zero_grad(set_to_none=True)

  def split_params(self, model):
    if len(self.muon_params) != 0:
      return

    for name, param in model.named_parameters():
      if not param.requires_grad:
        continue

      # checks to discriminate params
      is_gpt_hidden_weight = (
        name.startswith("gpt.gpt_layers.")
        and name.endswith(".weight")
        and param.ndim == 2
        and "layer_norm" not in name
      )
      is_bias = name.endswith(".bias")
      is_norm = "layer_norm" in name
      is_embedding = (
        name == "gpt.word_embedding.weight"
        or name == "gpt.pos_embedding.weight"
      )

      # separate params based on checks
      if is_gpt_hidden_weight:
        self.muon_params.append(param)
        self.muon_names.append(name)
      elif is_bias or is_norm or is_embedding:
        self.adamw_no_decay_params.append(param)
        self.adamw_no_decay_names.append(name)
      else:
        # Mainly classifier.weight in the model
        self.adamw_decay_params.append(param)
        self.adamw_decay_names.append(name)




def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)
  
  print(f"Training Dataset Size: {len(train_dataloader.dataset)}")
  print(f"Validation Dataset Size: {len(dev_dataloader.dataset)}")
  
  # Init model
  config = {
      'hidden_dropout_prob': args.hidden_dropout_prob,
      'num_labels': num_labels,
      'hidden_size': 768,
      'data_dir': '.',
      'fine_tune_mode': args.fine_tune_mode
  }

  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  # Set Optimizer
  #lr = args.lr
  #optimizer = AdamW(model.parameters(), lr=lr)
  optimizer = OptimWrapper(model, args)

  with mlflow.start_run(run_name=None) as run:
    # Log training parameters
    mlflow.log_params(vars(args))

    mlflow.set_tags({
        "study": args.study,
        "ablation_factor": args.ablation_factor,
    })

    best_dev_acc = -1.0
    best_f1_macro = -1.0
    best_step = 0
    best_time_sec = 0.0

    global_step = 0
    training_start_time = time.perf_counter()   # Logged

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
      epoch_start_time = time.perf_counter()    # Logged

      model.train()
      train_loss = 0
      num_batches = 0

      for batch_idx, batch in enumerate(
        tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
      ):
        b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        trainable_params = get_trainable_model_params(model)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        step_start_time = time.perf_counter()

        # Forward Pass
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

        # Backward Pass
        loss.backward()

        # grad_norm_global
        grad_norm_global = calculate_global_grad_norm(trainable_params)

        # Save weights before optimizer update
        parameters_before_step = clone_parameters(trainable_params)

        # Weight Update
        optimizer.step()
        global_step += 1

        # Save weights after optimizer update
        parameters_after_step = clone_parameters(trainable_params)

        # update norm, update/weight ratio, grad-update cosine
        update_metrics = calculate_update_metrics(parameters_before_step, parameters_after_step)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # step_time_sec
        step_end_time = time.perf_counter()
        step_time_sec = step_end_time - step_start_time

        # gpu_memory_peak
        if torch.cuda.is_available():
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**2
        else:
            gpu_memory_peak = 0.0

        # Calculate Metrics
        train_loss += loss.item()
        num_batches += 1

        # Batch loss
        batch_loss = loss.item()

        # Batch Accuracy
        logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=1).flatten()
        y_true = b_labels.detach().cpu().numpy()
        batch_acc = accuracy_score(y_true, y_pred)

        # Batch F1
        batch_f1 = f1_score(y_true, y_pred, average='macro')    # MACRO?????????????????

        batch_metrics = {
            "batch/loss": batch_loss, 
            "batch/accuracy": batch_acc, 
            "batch/f1_macro": batch_f1,
            "batch/grad_norm_global": grad_norm_global,
            "batch/step_time_sec": step_time_sec,
            "batch/gpu_memory_peak_mb": gpu_memory_peak,
        }
        batch_metrics.update(update_metrics)

        # Log Batch Metrics
        mlflow.log_metrics(
            batch_metrics,
            step=epoch * len(train_dataloader) + batch_idx,
        )        

      epoch_end_time = time.perf_counter()
      epoch_time_sec = epoch_end_time - epoch_start_time  # Logged

      train_loss = train_loss / (num_batches)

      train_metrics = model_eval(train_dataloader, model, device, return_dict=True, flag="train")
      val_metrics = model_eval(dev_dataloader, model, device, return_dict=True, flag="val")

      if val_metrics["val/accuracy"] > best_dev_acc:
        best_dev_acc = val_metrics["val/accuracy"]
        # save_model(model, optimizer, args, config, args.filepath)
        # expects optimizer.state_dict()

      current_f1_macro = val_metrics["val/f1_macro"]
      if current_f1_macro > best_f1_macro:
          best_f1_macro = current_f1_macro
          best_step = global_step
          best_time_sec = time.perf_counter() - training_start_time   # Logged

      # Log Epoch Metrics
      epoch_metrics = {
        "system/epoch_time_sec": epoch_time_sec,
        "system/time_to_best_f1_macro": best_time_sec,
        "system/steps_to_best_f1_macro": best_step,
        "val/best_accuracy": best_dev_acc,
        "val/best_f1_macro": best_f1_macro
      }
      epoch_metrics.update(train_metrics)
      epoch_metrics.update(val_metrics)

      mlflow.log_metrics(
          epoch_metrics,
          step=epoch,
      )
      
      # Log checkpoint at the end of each epoch
      #mlflow.pytorch.log_model(model, name=f"checkpoint_{epoch}")

      #print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
  gc.collect()

  if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()


def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Caution: weights_only=False can lead to malicious code execution !!!!!!!
    # inserted as a patch to make it work
    saved = torch.load(args.filepath, weights_only=False)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    print(f"Test Dataset Size: {len(test_dataloader.dataset)}")
    
    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print('DONE DEV')

    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")



# Basic Sampling Utilities

def log_uniform(gen: np.random.Generator, low: float, high: float) -> float:
    """
    Sample from a log-uniform distribution.

    Use this for scale-sensitive hyperparameters such as:
    - learning rate
    - weight decay
    """
    assert low > 0
    assert high > low

    log_low = np.log10(low)
    log_high = np.log10(high)
    return float(10 ** gen.uniform(low=log_low, high=log_high, size=None))


def uniform(gen: np.random.Generator, low: float, high: float) -> float:
    """
    Sample from a uniform distribution.
    """
    assert high >= low
    return gen.uniform(low, high)


def choice(gen: np.random.Generator, values: List[Any]) -> Any:
    """
    Sample one value from a discrete list.
    """
    assert len(values) > 0
    return gen.choice(values)


def broad_adamw(seed, itr=10):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Broad",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="AdamW",
      lr=log_uniform(gen, 1e-6, 5e-4),
      weight_decay=choice(gen, [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]),
      beta_1=choice(gen, [0.9]),
      beta_2=choice(gen, [0.95, 0.98, 0.999]),
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)


def fine_adamw(seed, itr=10):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Fine",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="AdamW",
      lr=log_uniform(gen, 1e-5, 2.5e-4),
      weight_decay=choice(gen, [1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 1e-1, 5e-1]),
      beta_1=choice(gen, [0.9]),
      beta_2=choice(gen, [0.95, 0.95, 0.98]),
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)


def seed1_adamw(seed, itr=1):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Seed",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="AdamW",
      lr=3.948898417713758e-05,
      weight_decay=choice(gen, [1e-2]),
      beta_1=choice(gen, [0.9]),
      beta_2=choice(gen, [0.95]),
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)

def seed2_adamw(seed, itr=1):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Seed",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="AdamW",
      lr=2.5820235894477722e-05,
      weight_decay=choice(gen, [5e-1]),
      beta_1=choice(gen, [0.9]),
      beta_2=choice(gen, [0.95]),
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)


def seed3_adamw(seed, itr=1):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Seed",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="AdamW",
      lr=6.158461899076557e-05,
      weight_decay=choice(gen, [1e-1]),
      beta_1=choice(gen, [0.9]),
      beta_2=choice(gen, [0.95]),
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)



def broad_muon(seed, itr=10):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Broad",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="Muon",
      lr=log_uniform(gen, 1e-4, 5e-2),
      weight_decay=choice(gen, [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]),
      momentum=choice(gen, [0.90, 0.95, 0.99]),
      ns_steps=choice(gen, [3, 5, 7]),
      adam_lr=3.948898417713758e-05,
      adam_weight_decay=0.01,
      adam_beta_1=0.9,
      adam_beta_2=0.95,
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)



def broad_normuon(seed, itr=10):
  gen = np.random.Generator(np.random.PCG64(seed=seed))

  fine_tune_mode = 'full-model' # 'last-linear-layer'
  
  for itr_no in range(1, itr+1):
    config = SimpleNamespace(
      study="Broad",
      ablation_factor="None",
      seed=seed,
      filepath='sst-classifier.pt',
      optim="NorMuon",
      lr=log_uniform(gen, 1e-4, 5e-2),
      weight_decay=choice(gen, [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]),
      beta_1=choice(gen, [0.90, 0.95, 0.99]),
      beta_2=choice(gen, [0.95, 0.98, 0.99, 0.999]),
      ns_steps=choice(gen, [3, 5, 7]),
      adam_lr=3.948898417713758e-05,
      adam_weight_decay=0.01,
      adam_beta_1=0.9,
      adam_beta_2=0.95,
      eps=1e-8,
      use_gpu=True,
      epochs=10,
      batch_size=32,
      hidden_dropout_prob=0.3,
      train='data/ids-sst-train.csv',
      dev='data/ids-sst-dev.csv',
      test='data/ids-sst-test-student.csv',
      fine_tune_mode=fine_tune_mode,
      dev_out='predictions/' + fine_tune_mode + '-sst-dev-out.csv',
      test_out='predictions/' + fine_tune_mode + '-sst-test-out.csv'
    )

    print('Training Sentiment Classifier on SST...')
    train(config)

    print('Evaluating on SST...')
    test(config)





if __name__ == "__main__":
  seed = 11711
  other_seeds = [11711, 2345, 6266, 7773]
  seed_everything(seed)

  #broad_adamw(seed, itr=20)
  #fine_adamw(seed, itr=11)
  #for s in other_seeds:
  #  seed1_adamw(s, itr=1)
  #  seed2_adamw(s, itr=1)
  #  seed3_adamw(s, itr=1)
  
  # Fix AdamW params
  for s in other_seeds:
    broad_muon(seed, itr=15)
    broad_normuon(seed, itr=15)


  # What sort of graph to have for the ablation study?




