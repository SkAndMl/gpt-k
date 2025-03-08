import os
import torch
from model import GPTk 
from loss import CrossEntropyK 
from typing import Tuple, Optional, List
import tiktoken
import math
import json
import random

# Creating appropriate repository for Model finetuning results
# and loading pertinent tokenizer
model_name = "gpt2-medium"
# Loading tokenizer
tokenizer = tiktoken.get_encoding('gpt2')
log_dir = f'finetuning_{model_name}'
log_file = os.path.join(log_dir, 'finetuning.txt')
per_token_file = os.path.join(log_dir, 'per_token_loss.txt')

# Creating the directory if it doesn't exist
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# Reset logs if already existing
with open(log_file, 'w') as f: pass 
with open(per_token_file, 'w') as f: pass

# Step 2: The dataset loader. Created specifically for Aquarat
class ReasoningDataset:

    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int,
        k: int,
        pad_token_id: int = 50256,
        ignore_index: int = -100,
        max_length: Optional[int] = None,
        device: str = "cpu"
    ) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.k = k
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.device = device
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.inputs, self.targets = [], []
        for example in self.data:
            self.inputs.append(tokenizer.encode(self.format_input(example)))
            self.targets.append(tokenizer.encode(self.format_output(example)))
        
        self.current_pos = 0


    def format_input(self, example) -> str:
        """
        Function to load Aquarat examples

        Parameters:
            example: Sample in dataset
        
        Returns:
            Question + Options formatted text
        """
        instruction_text = (
            f"For the given question, think and generate an answer"
        )
        input_text = (
            f"\n\n### Question\n{example['question']}\nOptions: {example['options']}"
        )
        return instruction_text + input_text
    
    def format_output(self, example) -> str:
        """
        Function to load Aquarat expected output

        Parameters:
            example: Sample in dataset
        
        Returns:
            response_text: Rationale + Correct Answer
        """

        response_text = (
            f"\n\n### Rationale\n{example['rationale']}\n\n### Correct answer\n{example['correct']}"
        )
        return response_text


    def get_batch(self):
        """
        Function to load individual batches

        Returns:
            tuple of inputs and targets
        """
        # If current iteration doesn't have enough samples for next batch:
        # Reset to beginning
        if self.current_pos+self.batch_size>=len(self.inputs):
            self.current_pos = 0
        
        inputs, targets = self.collate_fn(
            self.inputs[self.current_pos:self.current_pos+self.batch_size],
            self.targets[self.current_pos:self.current_pos+self.batch_size]
        )
        # Updating current position
        self.current_pos += self.batch_size
        return inputs, targets


    def collate_fn(
        self,
        instructions: List[int],
        responses: List[int]
    ) -> Tuple[torch.Tensor]:
        """
        Function for teach-forced batch creation. Accomodation made 
        to account for last k token prediction

        Parameters:
            instructions: List containing instuctions
            responses: List containing response to instructions
        Returns:
            Tuple containing Input and Target tensors
        """
        
        batch = [inp+tgt for inp, tgt in zip(instructions, responses)]
        # Max sequence length batch item for padding the rest 
        batch_max_length = max(len(item)+1 for item in batch)
        inputs_lst, targets_lst = [], []
        # Creating batch based on teacher-forcing format
        for i in range(len(batch)):
            new_item = batch[i].copy()
            new_item += [self.pad_token_id]

            padded = (
                new_item + [self.pad_token_id]*(batch_max_length - len(new_item) + self.k)
            )
            # Will usually be till -1; but till -k to account for extra k
            # token prediction
            inputs = torch.tensor(padded[:-self.k])
            # target offset by 1 for teacher-forcing
            targets = torch.tensor(padded[1:])
            # Setting ignore index for loss to not focus on the tokens 
            # in the instruction and padding
            targets[:len(instructions[i])-1] = self.ignore_index
            # Setting mask such that it corresponds to the padded id
            # in the target
            mask = targets == self.pad_token_id
            indices = torch.nonzero(mask).squeeze()
            # Other than first padding token, convert rest to ignore_index
            if indices.numel() > 1:
                targets[indices[1:]] = self.ignore_index
            # Inputs and targets cropped to max sequence length -- GPT2 limitation
            if self.max_length is not None:
                inputs = inputs[:self.max_length]
                targets = targets[:self.max_length+self.k-1]
            
            inputs_lst.append(inputs)
            targets_lst.append(targets)
        # Creating batch tensor
        inputs_tensor = torch.stack(inputs_lst).to(self.device)
        targets_tensor = torch.stack(targets_lst).to(self.device)

        return inputs_tensor, targets_tensor


# Parameters for training
batch_size = 32
grad_accum_steps = 8
freeze_steps = 1500
warmup_steps = 2000
eval_steps = 250
eval_loss_steps = 30
max_steps = 24360
max_length = 1024
min_lr, max_lr = 6e-5, 6e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
k, r = 3, 256
grad_norm_clip = 1.0


def get_lr(it):
    """
    Function for custom learning rate, based on the iteration

    Parameters:
        it : Current Iteration
    Returns:
        Learning Rate based on Heuristics 
    """
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

# Loading train and test dataset
train_ds = ReasoningDataset(
    data_path='data/train.jsonl',
    tokenizer=tokenizer,
    max_length=max_length,
    device=device,
    k=k,
    batch_size=batch_size
)
test_ds = ReasoningDataset(
    data_path="data/test.jsonl",
    tokenizer=tokenizer,
    max_length=max_length,
    device=device,
    k=k,
    batch_size=batch_size
)

# Loading GPTk model based on k and r parameters
print(f"Loading {model_name} with {k} LM heads and latent size {r}...")
model = GPTk.from_pretrained(model_name, k=k, r=r)
model.to(device)

# Freezing model backbone parameters
for param in model.parameters():
    param.requires_grad = False
# Fine-tuning only LM heads
for param in model.lm_heads.parameters():
    param.requires_grad = True

print("Initially, only LM heads are trainable.")

optimizer = torch.optim.AdamW([
    {"name": "frozen_block", "params": model.transformer['h'][:-6].parameters(), 'lr': 0.0},
    {"name": "last_6_blocks", "params": model.transformer['h'][-6:].parameters(), 'lr': 0.0},
    {"name": "latent_vector", "params": model.latent_layer.parameters(), 'lr': min_lr},
    {"name": "lm_heads", "params": model.lm_heads.parameters(), 'lr': min_lr}
],
    weight_decay=0.01)

# Custom Cross Entropy for GPTk
loss_fn = CrossEntropyK(k=k)


@torch.inference_mode()
def evaluate(model: GPTk):
    """
    Function to test model after training. Random samples are tried and
    output is presented in human-readable format

    Parameters:
        model: Model to test
    """
    model.eval()
    random_idxs = random.sample(
        list(range(len(test_ds))),
        k=3
    )

    for idx in random_idxs:
        example = test_ds.data[idx]
        input_text = test_ds.format_input(example)

        input_tokens = tokenizer.encode(input_text)
        input_tokens = torch.tensor([input_tokens]).to(device)

        output = model.generate(
            tokens=input_tokens
        )

        print(input_text)
        print(tokenizer.decode(output))
        print("="*100)

    model.train()

# The training Loop
for global_step in range(max_steps):
    # Test set validation done if current epoch multiple of eval_steps
    # or current_epoch == final_epoch
    if (global_step+1)%eval_steps==0 or global_step==max_steps-1:

        model.eval()
        with torch.inference_mode():
            total_val_loss = 0
            for _ in range(eval_loss_steps):
                x, y = test_ds.get_batch()
                logits = model(x)
                val_loss, per_token_loss_dict = loss_fn(logits=logits, targets=y)
                total_val_loss += val_loss.detach()
            
            total_val_loss /= eval_loss_steps
        # Logging Validation results
        to_log = f'val | step: {global_step:5d} | val_loss: {total_val_loss.item():.5f}'
        with open(log_file, 'a') as f: f.write(to_log+'\n')
        print(to_log)

        per_token_log = f'val | ' + f' | '.join([f'{i+1}:{v:.5f}' for i,v in per_token_loss_dict.items()])
        with open(per_token_file, 'a') as f: f.write(per_token_log+'\n')
        # If final epoch, save the weights
        if global_step==max_steps-1:
            checkpoint_path = os.path.join(log_dir, f"cpt.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': global_step,
                'val_loss': total_val_loss.item()
            }
            torch.save(checkpoint, checkpoint_path)
        
        evaluate(model)

        model.train()

    # The actual training loop
    loss_accum = 0
    optimizer.zero_grad()
    # Gradient accumulation across steps
    for _ in range(grad_accum_steps):
        x, y = train_ds.get_batch()
        logits = model(x)
        loss, per_token_loss_dict = loss_fn(logits=logits, targets=y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # Setting current learning rates
    lr = get_lr(global_step)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'frozen_block': continue
        # Last 6 blocks get updates only after current epoch == freeze step
        elif param_group['name'] == 'last_6_blocks': 
            if global_step >= freeze_steps:
                param_group['lr'] = lr
        else:
            param_group['lr'] = lr
    # Clipping Gradient for normalization during weight update
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
    optimizer.step()
    # Logging training parameters
    to_log = f'train | step: {global_step:5d} | train_loss: {loss_accum.item():.5f} | lr: {lr:.4e} | norm: {norm:.4f}'
    with open(log_file, 'a') as f: f.write(to_log+'\n')
    print(to_log)

    per_token_log = f'train | ' + f' | '.join([f'{i+1}:{v:.5f}' for i,v in per_token_loss_dict.items()])
    with open(per_token_file, 'a') as f: f.write(per_token_log+'\n')