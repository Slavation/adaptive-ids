# **Reptile-Based Few-Shot Intrusion Detection System**
#  Modified the adaptive-ids.ipynb file to allow user input for easier experimentation
#  @Author: Vladislav Zagidulin (2026)
#  @Supervisor: Prof. Marwa Elsayed
#
# Based on the Reptile meta-learning algorithm proposed by Alex Nichol and John Schulman [[1]](https://openai.com/research/reptile) and the Torch Reptile implementation by Ruduan B. F. Plug [[2]](https://github.com/dualslash/reptile-torch)</font>

# Meta Libraries
# System Utility
import sys
print(sys.version)

# Packages
# Data Processing
import numpy as np

# Parallel Compute
import torch 
import torch.nn as nn

# Utility Libraries
from copy import deepcopy
import hashlib
import tqdm

# Initialize Device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("Torch Version\t", torch.__version__)

# Chose which attacks to test model on
# Usage: python adaptive-ids.py "Attack1" "Attack2"
try:
    target_a1 = sys.argv[1]
    target_a2 = sys.argv[2]
except IndexError:
    # Default values for testing
    target_a1 = "Backdoor"
    target_a2 = "DDoS_HTTP"
    print("Input not accepted, using default attacks for testing.")

# Generate reproducible seed based on the attack names
seed_str = f"{min(target_a1, target_a2)}_{max(target_a1, target_a2)}"
seed_val = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

print(f"Executing: {target_a1} vs {target_a2}")
print(f"Deterministic Seed: {seed_val}")

# Apply seeds to all libraries
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# Meta-Learning Framework
# Reptile Class Definition

from sklearn.metrics import precision_score, recall_score, f1_score

class Reptile:

  def __init__(self, model, reptile_params):

    # Intialize Reptile Parameters
    self.inner_step_size = reptile_params[0]
    self.inner_steps = reptile_params[1]
    self.outer_step_size = reptile_params[2]
    self.outer_steps = reptile_params[3]
    self.meta_batch_size = reptile_params[4]
    self.criterion = nn.CrossEntropyLoss()

    # Initialize Torch Model
    self.model = model.to(device)

  def reset(self):
    # Reset Training Gradients
    self.model.zero_grad()

  def train(self, x, y):

    # Reset gradients
    self.reset()
    self.model.train()

    # Outer Training Loop
    for outer_iteration in tqdm.tqdm(range(self.outer_steps)):

      # Track Current Weights
      current_weights = deepcopy(self.model.state_dict())

      # Sample a new Subtask
      (x_support, y_support), _ = sample_task(
         x, y,
         allowed_classes=train_classes,
         n_way=N_WAY,
         k_shot=K_SHOT,
         query_size=QUERY_SIZE
      )

      x_support = x_support.to(device)
      y_support = y_support.to(device)

      # Inner Training Loop
      for inner_iteration in range(self.inner_steps):

        perm = torch.randperm(x_support.shape[0])

        # Process Meta Learning Batches
        for batch in range(0, x_support.shape[0], self.meta_batch_size):

          # Get Permuted Batch from Sample
          idx = perm[batch:batch + self.meta_batch_size]

          # Calculate Batch Loss
          self.model.zero_grad()
          batch_loss = self.loss(x_support[idx], y_support[idx])
          batch_loss.backward()

          # Update Model Parameters
          with torch.no_grad():
            for theta in self.model.parameters():
                if theta.grad is not None:
                    theta.add_(theta.grad, alpha=-self.inner_step_size)

      # Linear Cooling Schedule
      alpha = self.outer_step_size * (1 - outer_iteration / self.outer_steps)

      # Get Current Candidate Weights
      candidate_weights = self.model.state_dict()

      # Transfer Candidate Weights to Model State Checkpoint
      state_dict = {candidate: (current_weights[candidate] + alpha *
                               (candidate_weights[candidate] - current_weights[candidate]))
                                for candidate in candidate_weights}
      self.model.load_state_dict(state_dict)

  def loss(self, x, y):
    logits = self.model(x)
    return self.criterion(logits, y)

  def predict(self, x):
    self.model.eval()

    with torch.no_grad():
        # Convert only if needed
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)

        # Ensure correct device and dtype
        t = t.to(device=device, dtype=torch.float32)

        logits = self.model(t)
        prediction = torch.argmax(logits, dim=1)

    return prediction.cpu().numpy()

  def eval(self, x, y, allowed_classes, gradient_steps=5):
      self.model.train()

      # Sample a task
      (x_support, y_support), (x_query, y_query) = sample_task(
         x, y,
         allowed_classes=allowed_classes,
         n_way=N_WAY,
         k_shot=K_SHOT,
         query_size=QUERY_SIZE
      )

      x_support = x_support.to(device)
      y_support = y_support.to(device)
      x_query   = x_query.to(device)
      y_query   = y_query.to(device)

      # Store Meta-Initialization Weights
      meta_weights = deepcopy(self.model.state_dict())

      # Calculate Estimate over Gradient Steps
      for step in range(gradient_steps):

        # Calculate Evaluation Loss and Backpropagate
        self.model.zero_grad()
        loss = self.loss(x_support, y_support)
        loss.backward()

        # Update Model Estimate Parameters
        with torch.no_grad():
          for theta in self.model.parameters():
              if theta.grad is not None:
                  theta.add_(theta.grad, alpha=-self.inner_step_size)

      # Get Estimate Loss over Evaluation
      self.model.eval()
      with torch.no_grad():
          logits = self.model(x_query)
          predictions = torch.argmax(logits, dim=1)

      # Accuracy
      accuracy = (predictions == y_query).float().mean().item()

      # Precision
      precision = precision_score(
          y_query.cpu().numpy(),
          predictions.cpu().numpy(),
          average='macro',
          zero_division=0
      )

      # Recall
      recall = recall_score(
          y_query.cpu().numpy(),
          predictions.cpu().numpy(),
          average='macro',
          zero_division=0
      )

      # F1 Score
      f1 = f1_score(
          y_query.cpu().numpy(),
          predictions.cpu().numpy(),
          average='macro',
          zero_division=0
      )

      # Restore Meta-Initialization Weights
      self.model.load_state_dict(meta_weights)
      self.model.train()

      return accuracy, precision, recall, f1

def evaluate_episodes(model, x, y, allowed_classes, n_episodes, gradient_steps=5):
    accs, precs, recs, f1s = [], [], [], []
    for _ in range(n_episodes):
        acc, prec, rec, f1 = model.eval(x, y, allowed_classes=allowed_classes, gradient_steps=gradient_steps)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return (
        np.mean(accs), np.std(accs),
        np.mean(precs), np.std(precs),
        np.mean(recs), np.std(recs),
        np.mean(f1s), np.std(f1s)
    )

# Task Sampler
N_WAY = 3       # number of classes per episode
K_SHOT = 15      # support samples per class
QUERY_SIZE = 15  # query samples per class
HIDDEN_DIM = 128

def sample_task(x, y, allowed_classes, n_way=N_WAY, k_shot=K_SHOT, query_size=QUERY_SIZE, return_classes=False):
    # Work with numpy view of labels
    y_np = y.cpu().numpy()
    needed = k_shot + query_size

    # Only classes that are allowed and have enough samples
    valid_classes = [
        c for c in allowed_classes
        if np.sum(y_np == c) >= needed
    ]

    if len(valid_classes) < n_way:
        raise ValueError("Not enough valid classes for this N-way task")

    # Randomly choose n_way classes from the valid ones
    classes = np.random.choice(valid_classes, n_way, replace=False)

    support_x = []
    support_y = []
    query_x = []
    query_y = []

    # Map global label -> episodic label 0..n_way-1
    class_mapping = {int(c): i for i, c in enumerate(classes)}

    for c in classes:
        idx = np.where(y_np == c)[0]             # indices of this class
        chosen = np.random.choice(idx, needed, replace=False)

        support = chosen[:k_shot]
        query = chosen[k_shot:]

        support_x.append(x[support])
        query_x.append(x[query])

        # Episodic labels 0..n_way-1
        episodic_label = class_mapping[int(c)]
        support_y.append(torch.full((len(support),), episodic_label, dtype=torch.long))
        query_y.append(torch.full((len(query),), episodic_label, dtype=torch.long))

    x_support = torch.cat(support_x, dim=0)
    y_support = torch.cat(support_y, dim=0)
    x_query = torch.cat(query_x, dim=0)
    y_query = torch.cat(query_y, dim=0)

    if return_classes:
        return (x_support, y_support), (x_query, y_query), classes
    else:
        return (x_support, y_support), (x_query, y_query)

# PyTorch Module
class TorchModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_classes=N_WAY):
        super(TorchModule, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        x = torch.relu(self.input(x))
        features = torch.relu(self.hidden(x))
        logits = self.output(features)

        if return_features:
            return logits, features
        return logits

# Dataset

# Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("dataset/ML-EdgeIIoT-dataset-test.csv", low_memory=False)

# Remove duplicate entries
df.drop_duplicates(subset=None, keep="first", inplace=True)

# Keep the most important features
keep_cols = {
    "dns.qry.name",
    "dns.qry.name.len",
    "dns.retransmit_request",
    "http.request.version",
    "mqtt.conflags",
    "mqtt.len",
    "mqtt.msgtype",
    "mqtt.topic",
    "mqtt.topic_len",
    "mqtt.ver",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.len",
    "tcp.seq",
    "udp.time_delta",
    "Attack_type"
}

drop_cols = [c for c in df.columns if c not in keep_cols]
df.drop(columns=drop_cols, inplace=True)

# Clean values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)

# Encode categorical features
categorical_cols = [
    'http.request.version',
    'mqtt.topic',
    "dns.qry.name.len",
]

df = pd.get_dummies(df, columns=categorical_cols)

# Separate features and labels
x_raw = df.drop(columns=["Attack_type"]).values

# Encode Attack_type labels
encoder = LabelEncoder()
y_raw = encoder.fit_transform(df["Attack_type"])

print("Dataset loaded:", x_raw.shape, y_raw.shape)

# Split Data

# Get all unique class labels
all_classes = np.unique(y_raw)

# Labels of normal traffic and chosen test attacks from attack traffic
normal_label = encoder.transform(['Normal'])[0]
a1_label = encoder.transform([target_a1])[0]
a2_label = encoder.transform([target_a2])[0]

attack_classes = [c for c in all_classes if c != normal_label]

# Force chosen attack into test set. Keeps an 80/20 train/test split
test_attacks = [a1_label, a2_label]
train_attacks = [c for c in attack_classes if c not in test_attacks]

# Split normal class by rows to ensure no leakage into test set
normal_idx = np.where(y_raw == normal_label)[0]
np.random.shuffle(normal_idx)

cut = int(0.8 * len(normal_idx))
normal_train_idx = normal_idx[:cut]
normal_test_idx  = normal_idx[cut:]

# Get indices for the chosen train/test attack types
attack_train_idx = np.where(np.isin(y_raw, train_attacks))[0]
attack_test_idx  = np.where(np.isin(y_raw, test_attacks))[0]

# Combine
train_idx = np.concatenate([attack_train_idx, normal_train_idx])
test_idx  = np.concatenate([attack_test_idx,  normal_test_idx])

# Shuffle indices
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

# Apply split
x_train_np = x_raw[train_idx]
y_train_np = y_raw[train_idx]

x_test_np  = x_raw[test_idx]
y_test_np  = y_raw[test_idx]

# Initialize scaler to standardize features
scaler = StandardScaler()

# Fit on train set and transform both train and test sets
x_train_scaled = scaler.fit_transform(x_train_np)
x_test_scaled = scaler.transform(x_test_np)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)

x_test  = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test  = torch.tensor(y_test_np, dtype=torch.long)

# Define final classes
train_classes = np.unique(y_train_np)
test_classes  = np.unique(y_test_np)

print("The model will learn these classes:")
print(encoder.inverse_transform(train_classes))

print("\nThe model will be tested on these classes:")
print(encoder.inverse_transform(test_classes))

# Experiment
import os
import csv

# Define Experiment Parameters
inner_step_size = 0.02
inner_steps = 15

outer_step_size = 0.1
outer_iterations = 2000
meta_batch_size = 15

params = [inner_step_size, inner_steps,
          outer_step_size, outer_iterations, meta_batch_size]

# Build Model
n_features = x_train.shape[1]
reptile_model = Reptile(TorchModule(n_features, hidden_dim=HIDDEN_DIM, num_classes=N_WAY), params)

# Train Model
reptile_model.train(x_train, y_train)

# Zero-day evaluation on unseen test classes
mean_acc, std_acc, mean_prec, std_prec, mean_rec, std_rec, mean_f1, std_f1 = evaluate_episodes(
    reptile_model,
    x_test, y_test,
    allowed_classes=test_classes,
    n_episodes=200,
)

ci_acc  = 1.96 * std_acc  / np.sqrt(200)
ci_prec = 1.96 * std_prec / np.sqrt(200)
ci_rec = 1.96 * std_rec / np.sqrt(200)
ci_f1 = 1.96 * std_f1 / np.sqrt(200)

print(f"Zero-Day {N_WAY}-Way Accuracy : {mean_acc:.4f} ± {ci_acc:.4f}")
print(f"Zero-Day {N_WAY}-Way Precision: {mean_prec:.4f} ± {ci_prec:.4f}")
print(f"Zero-Day {N_WAY}-Way Recall: {mean_rec:.4f} ± {ci_rec:.4f}")
print(f"Zero-Day {N_WAY}-Way F1-Score: {mean_f1:.4f} ± {ci_f1:.4f}")

# Write results to csv file
results_file = "results_15shot.csv"
file_exists = os.path.isfile(results_file)

with open(results_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["Attack_1", "Attack_2", "Seed", "F1", "Precision", "Recall"])
    if not file_exists:
        writer.writeheader()

    writer.writerow({
        "Attack_1": target_a1,
        "Attack_2": target_a2,
        "Seed": seed_val,
        "F1": round(mean_f1, 4),
        "Precision": round(mean_prec, 4),
        "Recall": round(mean_rec, 4)
    })

print(f"Results appended for {target_a1} vs {target_a2}")