"""
Adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        """
        Initializing the self attention layer
        
        Parameters:
            config: Config file with parameters
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Same head used for Q, K and V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # FFN for embeddings
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, x):
        # B -- batch size, T -- sequence length, C - embedding dimension
        B, T, C = x.size() 
        # Q, K, V projection
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # is_causal: masked self attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # FFN
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        """
        Initializing MLP 

        Parmaters:
         config: Config file with parameters
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # Projection to higher dim
        x = self.c_fc(x)
        # Activation
        x = self.gelu(x)
        # Reduction to lower dim
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        """
        Initializing transformer block

        Parameters:
            config: Config file with the parameters
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Self attention based on above class
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # MLP projection
        self.mlp = MLP(config)

    def forward(self, x):
        # Skip connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 24 # number of layers
    n_head: int = 16 # number of heads
    n_embd: int = 1024 # embedding dimension
    k: int = 3
    r: int = 256


class LMHead(nn.Module):

    def __init__(self, vocab_size, r): 
        """
        Initializing Language Model Head

        Parameters:
            vocab_size: Size of vocab for logits
            r: Latent dim
        """
        super().__init__()
        self.up = nn.Linear(r, vocab_size)
        self.down = nn.Linear(vocab_size, r)

    def forward(self, x: torch.Tensor):
        # Calculating logits
        logits = self.up(x)
        # Getting back the latent
        output = self.down(F.relu(logits))
        return logits, output


class GPTk(nn.Module):
    def __init__(self, config: GPTConfig):
        """
        Initializing GPTk

        Parameters:
            config: Config file with parameters
        """
        super().__init__()
        self.config = config
        # The transformer block in GPT
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            ln_f = nn.LayerNorm(config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        ))
        # The modifications: Latent layer and Language Model head for K
        self.latent_layer = nn.Linear(config.n_embd, config.r)
        self.lm_heads = nn.ModuleList([LMHead(self.config.vocab_size, self.config.r)
                                       for _ in range(self.config.k)])
        # Weight initializations based on GPT paper
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))            

    def get_num_params(self, non_embedding=True):
        """
        Function for returning parameter count

        Parameters:
            non_embedding: Boolean to signify if embedding weights to be ignored
        Returns:
            n_params: Parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Function to initialize weights in GPT
        
        Parameters:
            module: The model object
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
        
        
    def forward(self, idx):
        device = idx.device
        # Checking sequence length size and asserting sequence length size
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # Positional embedding tensor 
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # Creating token embedding and positional embedding
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        # Input: token_embedding + positional_embedding
        x = tok_emb + pos_emb
        # Input processed through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # Layer norm
        x = self.transformer.ln_f(x)
        # Curating the latent vector
        latent_vector = self.latent_layer(x)
        # Processing outputs for GPTk using latent vector
        logits = {} 
        lm_head_op = None       
        for k in range(len(self.lm_heads)):
            if k==0:
                logits[k], lm_head_op = self.lm_heads[k](latent_vector)
            else:
                logits[k], lm_head_op = self.lm_heads[k](latent_vector + lm_head_op)
        # Returning k logits
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type="gpt2", k: int=3, r: int = 64):
        """
        Function to load pre-trained weights for GPTk
        
        Parameters:
            cls: GPTk class
            model_type: Type of GPT backbone
            k: GPT k parameters
            r: Latent Vector dimension
        
        Returns:

        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"Loading weights from pretrained GPT-2 model: {model_type}")
        # Creating config file based on chosen backbone
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # Adding vocab size and input length to config
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # Loading the backbone
        model_hf = GPT2LMHeadModel.from_pretrained(f"./{model_type}")
        model_hf.eval()
        # Freezing backbone parameters
        for param in model_hf.parameters():
            param.requires_grad = False
        print("Frozen all transformer weights.")
        # Deleting existing Language Model Head
        if hasattr(model_hf, "lm_head"):
            del model_hf.lm_head
            print("Original LM head removed.")
        # Alinging model weights; 
        print("Aligning model weights...")
        config_args.update({"k": k, "r": r})
        config = GPTConfig(**config_args)
        # Loading GPTk model
        model = cls(config)  
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # assert len(sd_keys_hf) == len(sd_keys), f"Mismatch: {len(sd_keys_hf)} vs {len(sd_keys)}"
        # Transferring weight from backbone to GPTk
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Modified {model_type} model loaded into GPTk.")
        return model

    
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 512
    ) -> None:
        
        input_len = tokens.shape[0]

        for _ in range(max_new_tokens):
            logits = self(tokens)
            next_tokens, next_token_probs = [], []
            for i in range(self.config.k):
                next_tokens.append(torch.argmax(logits[i][:, -1, :], dim=-1).squeeze().item())
                max_val, _ = torch.max(F.softmax(logits[i], dim=-1), dim=-1)
                next_token_probs.append(max_val[0][0].item())
            
            tokens = torch.cat(
                [tokens, torch.tensor([[next_tokens[0]]], device=tokens.device)], 
                dim=-1
            ).to(tokens.device)
            if next_tokens[0]==50256: break
        
        return tokens[0].tolist()[input_len:]
    


class GPTv(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            ln_f = nn.LayerNorm(config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.lm_vectors = nn.Embedding(num_embeddings=config.k, embedding_dim=config.n_embd)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))            

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        
        logits = {} 
        for k in range(self.config.k):
            logits[k] = self.lm_head(
                self.transformer.ln_f(x + self.lm_vectors(torch.tensor(k, device=x.device)))
            )

        return logits
    
    @classmethod
    def from_pretrained(cls, model_type="gpt2", k: int=3, r: int = 64):

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"Loading weights from pretrained GPT-2 model: {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        model_hf = GPT2LMHeadModel.from_pretrained(f"./{model_type}")
        model_hf.eval()

        for param in model_hf.parameters():
            param.requires_grad = False
        print("Frozen all transformer weights.")

        print("Aligning model weights...")
        config_args.update({"k": k, "r": r})
        config = GPTConfig(**config_args)
        model = cls(config)  
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # assert len(sd_keys_hf) == len(sd_keys), f"Mismatch: {len(sd_keys_hf)} vs {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Modified {model_type} model loaded into GPTk.")
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 512
    ) -> None:
        
        input_len = tokens.shape[0]

        for _ in range(max_new_tokens):
            logits = self(tokens)
            next_tokens, next_token_probs = [], []
            for i in range(self.config.k):
                next_tokens.append(torch.argmax(logits[i][:, -1, :], dim=-1).squeeze().item())
                max_val, _ = torch.max(F.softmax(logits[i], dim=-1), dim=-1)
                next_token_probs.append(max_val[0][0].item())
            
            tokens = torch.cat(
                [tokens, torch.tensor([[next_tokens[0]]], device=tokens.device)], 
                dim=-1
            ).to(tokens.device)
            if next_tokens[0]==50256: break
        
        return tokens[0].tolist()[input_len:]