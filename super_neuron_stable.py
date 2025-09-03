import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import json
import pickle
import os
from collections import Counter
import re


class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        words = []
        for text in texts:
            words.extend(self.tokenize(text))
        
        word_counts = Counter(words)
        
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        
        for word, _ in word_counts.most_common():
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        
    def tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())
    
    def encode(self, text, max_length=None):
        tokens = self.tokenize(text)
        if max_length:
            tokens = tokens[:max_length-2]
        
        token_ids = [self.word_to_idx.get("<START>", 2)]
        for token in tokens:
            token_ids.append(self.word_to_idx.get(token, 1))
        token_ids.append(self.word_to_idx.get("<END>", 3))
        
        if max_length:
            while len(token_ids) < max_length:
                token_ids.append(0)
                
        return token_ids
    
    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.idx_to_word.get(idx, "<UNK>")
            if word == "<END>":
                break
            if word not in ["<PAD>", "<START>"]:
                words.append(word)
        return " ".join(words)
    
    def save_tokenizer(self, filepath):
        """Sauvegarde le tokenizer"""
        tokenizer_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer sauvegardé: {filepath}")
    
    @classmethod
    def load_tokenizer(cls, filepath):
        """Charge un tokenizer sauvegardé"""
        tokenizer = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        tokenizer.word_to_idx = data['word_to_idx']
        tokenizer.idx_to_word = data['idx_to_word']
        tokenizer.vocab_size = data['vocab_size']
        print(f"Tokenizer chargé: {filepath}")
        return tokenizer


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialisation stable
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
    def forward(self, query, key, value):
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key.size(1)
        
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, d_model
        )
        
        output = self.W_o(attention_output)
        return output


class SuperNeuronStable(nn.Module):
    def __init__(self, d_model, num_branches, num_heads_per_branch):
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        
        self.branches = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads_per_branch, dropout=0.1) 
            for _ in range(num_branches)
        ])
        
        # Router qui apprend quels tokens vont dans quelles branches
        self.router = nn.Linear(d_model, num_branches)
        self.combiner = nn.Linear(d_model, d_model)
        
    def forward(self, x, use_routing=True):
        batch_size, seq_len, d_model = x.size()
        
        if use_routing:
            # Calcul des scores de routage
            router_scores = self.router(x)  # [batch, seq_len, num_branches]
            router_probs = F.softmax(router_scores, dim=-1)
            
            # Calcul de chaque branche
            branch_outputs = []
            for i, branch in enumerate(self.branches):
                branch_out = branch(x, x, x)  # [batch, seq_len, d_model]
                branch_outputs.append(branch_out)
            
            # Combinaison pondérée des sorties
            combined = torch.zeros_like(x)
            for i, branch_out in enumerate(branch_outputs):
                weights = router_probs[:, :, i:i+1]  # [batch, seq_len, 1]
                combined += weights * branch_out
        else:
            # Mode simple: moyenne de toutes les branches
            branch_outputs = []
            for branch in self.branches:
                branch_out = branch(x, x, x)
                branch_outputs.append(branch_out)
            combined = sum(branch_outputs) / len(branch_outputs)
        
        output = self.combiner(combined)
        return output


class SuperNeuronLM(nn.Module):
    def __init__(self, vocab_size, d_model=96, num_branches=4, num_heads_per_branch=6, max_seq_len=48):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, d_model))
        
        self.super_neuron = SuperNeuronStable(d_model, num_branches, num_heads_per_branch)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x, use_routing=False):
        seq_len = x.size(1)
        
        embeddings = self.embedding(x)
        embeddings = embeddings + self.positional_encoding[:seq_len, :]
        embeddings = self.dropout(embeddings)
        
        attention_output = self.super_neuron(embeddings, use_routing=use_routing)
        
        output = self.layer_norm(attention_output)
        logits = self.output_proj(output)
        
        return logits
    
    def save_model(self, filepath):
        """Sauvegarde le modèle complet"""
        model_data = {
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_branches': self.super_neuron.num_branches,
                'num_heads_per_branch': self.super_neuron.branches[0].num_heads,
                'max_seq_len': self.positional_encoding.size(0)
            }
        }
        torch.save(model_data, filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """Charge un modèle sauvegardé"""
        model_data = torch.load(filepath, map_location=device)
        config = model_data['config']
        
        model = cls(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_branches=config['num_branches'],
            num_heads_per_branch=config['num_heads_per_branch'],
            max_seq_len=config['max_seq_len']
        )
        
        model.load_state_dict(model_data['state_dict'])
        model.eval()
        print(f"Modèle chargé: {filepath}")
        return model


def load_dataset():
    try:
        with open('/home/payet/projet/extended_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Le fichier a une structure avec "training_texts"
        if 'training_texts' in data:
            return data['training_texts']
        # Sinon, essaie la structure classique avec des objets
        elif isinstance(data, list):
            return [item.get('text', '') for item in data if 'text' in item]
        else:
            return []
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {e}")


def create_training_pairs(texts, tokenizer, max_length=24):
    pairs = []
    
    for text in texts:
        token_ids = tokenizer.encode(text, max_length)
        
        for i in range(len(token_ids) - 1):
            input_seq = token_ids[:i+1]
            target = token_ids[i+1]
            
            if len(input_seq) < max_length:
                input_seq = input_seq + [0] * (max_length - len(input_seq))
            
            pairs.append((input_seq, target))
    
    return pairs


def train_model(model, train_pairs, tokenizer, epochs=30, batch_size=8, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_pairs)
        num_batches = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            
            inputs = torch.tensor([pair[0] for pair in batch])
            targets = torch.tensor([pair[1] for pair in batch])
            
            optimizer.zero_grad()
            
            logits = model(inputs, use_routing=False)
            loss = criterion(logits[:, -1, :], targets)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")


def generate_text(model, tokenizer, prompt, max_length=24, temperature=0.7):
    model.eval()
    
    token_ids = tokenizer.encode(prompt, max_length=24)
    input_tensor = torch.tensor([token_ids])
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor, use_routing=True)  # Utilise le routage en inférence
            next_token_logits = logits[0, -1, :] / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == tokenizer.word_to_idx.get("<END>", 3):
                break
                
            generated.append(next_token)
            
            new_token = torch.tensor([[next_token]])
            input_tensor = torch.cat([input_tensor, new_token], dim=1)
            
            if input_tensor.size(1) >= 24:
                input_tensor = input_tensor[:, 1:]
    
    return tokenizer.decode(generated)


if __name__ == "__main__":
    print("=== SUPER NEURON LANGUAGE MODEL STABLE ===")
    
    print("1. Chargement des données...")
    texts = load_dataset()
    print(f"Nombre de textes: {len(texts)}")
    
    print("2. Construction du vocabulaire...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    print(f"Taille du vocabulaire: {tokenizer.vocab_size}")
    
    print("3. Création du modèle...")
    model = SuperNeuronLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_branches=8,
        num_heads_per_branch=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux: {total_params:,}")
    
    print("4. Préparation des données...")
    train_pairs = create_training_pairs(texts, tokenizer)
    print(f"Nombre de paires d'entraînement: {len(train_pairs)}")
    
    print("5. Entraînement...")
    train_model(model, train_pairs, tokenizer, epochs=50)
    
    # 6. Sauvegarde du modèle et tokenizer
    print("6. Sauvegarde...")
    os.makedirs('saved_models', exist_ok=True)
    model.save_model('saved_models/super_neuron_model.pth')
    tokenizer.save_tokenizer('saved_models/tokenizer.pkl')
    
    print("7. Test de génération:")
    prompts = ["Les algorithmes", "L'intelligence artificielle", "Les réseaux de neurones", "Le traitement du langage"]
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=24)
        print(f"'{prompt}' -> '{generated}'")
    
    print("\nEntraînement terminé !")
