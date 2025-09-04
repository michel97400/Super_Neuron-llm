#!/usr/bin/env python3
"""
Script d'entraînement optimisé avec métriques, early stopping et architecture améliorée
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import math
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Import optionnel de matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib non disponible - graphiques désactivés")

# Imports locaux
from conversational_model import ConversationalLM
from optimized_architecture import OptimizedSuperNeuronConversational
from sequence_optimizer import SequenceOptimizer

class TrainingOptimizer:
    """Gestionnaire d'entraînement optimisé"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.metrics = defaultdict(list)
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def compute_metrics(self, outputs, targets, mask=None):
        """Calcule des métriques détaillées"""
        with torch.no_grad():
            # Loss principale
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), 
                                 reduction='none')
            
            if mask is not None:
                loss = loss * mask.view(-1)
                valid_tokens = mask.sum().item()
                loss = loss.sum() / valid_tokens
            else:
                loss = loss.mean()
            
            # Précision
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == targets)
            
            if mask is not None:
                correct = correct * mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = correct.float().mean()
            
            # Perplexité
            perplexity = torch.exp(loss)
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'perplexity': perplexity.item()
            }
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Entraîne une époque avec métriques"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            # Calcul des métriques
            metrics = self.compute_metrics(outputs, targets, mask)
            
            # Backward pass
            loss = torch.tensor(metrics['loss'], requires_grad=True, device=self.device)
            if loss.requires_grad:
                # Recalculer la loss pour le gradient
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                if mask is not None:
                    loss = loss * mask.view(-1)
                    loss = loss.sum() / mask.sum()
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Collecter métriques
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Mise à jour barre de progression
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'Acc': f"{metrics['accuracy']:.3f}",
                'PPL': f"{metrics['perplexity']:.2f}"
            })
        
        # Moyennes de l'époque
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def validate(self, dataloader):
        """Validation avec métriques"""
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                
                outputs = self.model(input_ids)
                metrics = self.compute_metrics(outputs, targets, mask)
                
                for key, value in metrics.items():
                    val_metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    def early_stopping_check(self, val_loss, patience=5):
        """Vérification early stopping"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False  # Continue training
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience
    
    def save_checkpoint(self, epoch, optimizer, path="checkpoint.pth"):
        """Sauvegarde complète"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'metrics': dict(self.metrics)
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint sauvegardé : {path}")
    
    def plot_metrics(self, save_path="training_metrics.png"):
        """Graphiques des métriques"""
        if not HAS_MATPLOTLIB:
            print("📊 matplotlib non disponible - graphiques ignorés")
            return
            
        if not self.metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Métriques d\'entraînement', fontsize=16)
        
        # Loss
        axes[0,0].plot(self.metrics['train_loss'], label='Train', color='blue')
        axes[0,0].plot(self.metrics['val_loss'], label='Validation', color='red')
        axes[0,0].set_title('Loss')
        axes[0,0].set_xlabel('Époque')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Accuracy
        axes[0,1].plot(self.metrics['train_accuracy'], label='Train', color='blue')
        axes[0,1].plot(self.metrics['val_accuracy'], label='Validation', color='red')
        axes[0,1].set_title('Précision')
        axes[0,1].set_xlabel('Époque')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Perplexity
        axes[1,0].plot(self.metrics['train_perplexity'], label='Train', color='blue')
        axes[1,0].plot(self.metrics['val_perplexity'], label='Validation', color='red')
        axes[1,0].set_title('Perplexité')
        axes[1,0].set_xlabel('Époque')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Learning rate (si disponible)
        if 'learning_rate' in self.metrics:
            axes[1,1].plot(self.metrics['learning_rate'], color='green')
            axes[1,1].set_title('Learning Rate')
            axes[1,1].set_xlabel('Époque')
            axes[1,1].grid(True)
        else:
            axes[1,1].text(0.5, 0.5, 'Learning Rate\nNon disponible', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Graphiques sauvegardés : {save_path}")

def create_optimized_dataloader(texts, llm, sequence_optimizer, batch_size=8, max_length=100):
    """Crée un dataloader optimisé"""
    
    # Optimiser les séquences
    print("🔄 Optimisation des séquences...")
    optimized_texts = sequence_optimizer.optimize_dataset(texts, llm)
    
    # Statistiques
    stats = sequence_optimizer.get_statistics(optimized_texts, llm)
    print(f"📊 Séquences optimisées :")
    print(f"   • Nombre : {stats['count']}")
    print(f"   • Longueur moy. : {stats['avg_length']:.1f} tokens")
    print(f"   • Ratio optimal : {stats['optimal_ratio']:.1%}")
    
    # Tokenization
    sequences = []
    for text in optimized_texts:
        encoded = llm.encode(text)
        if len(encoded) >= 5:  # Minimum viable
            sequences.append(encoded)
    
    # Créer batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        
        # Padding
        max_len = min(max_length, max(len(seq) for seq in batch_sequences))
        
        batch_input_ids = []
        batch_targets = []
        batch_masks = []
        
        for seq in batch_sequences:
            # Tronquer si nécessaire
            if len(seq) > max_len:
                seq = seq[:max_len]
            
            # Input et target (décalé de 1)
            input_ids = seq[:-1] + [llm.vocab['<pad>']] * (max_len - len(seq))
            targets = seq[1:] + [llm.vocab['<pad>']] * (max_len - len(seq) + 1)
            mask = [1] * (len(seq) - 1) + [0] * (max_len - len(seq) + 1)
            
            batch_input_ids.append(input_ids[:max_len-1])
            batch_targets.append(targets[:max_len-1])
            batch_masks.append(mask[:max_len-1])
        
        if batch_input_ids:  # Éviter les batches vides
            batches.append({
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'targets': torch.tensor(batch_targets, dtype=torch.long),
                'mask': torch.tensor(batch_masks, dtype=torch.float)
            })
    
    return batches

def main():
    print("🚀 === ENTRAÎNEMENT SUPER NEURON OPTIMISÉ ===")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device : {device}")
    
    # Charger les données
    print("\n📚 Chargement du dataset enrichi...")
    with open('conversational_dataset_enhanced.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = data['conversational_texts']
    print(f"✅ {len(texts)} textes chargés")
    
    # Préparer le tokenizer
    print("\n🔤 Construction du vocabulaire...")
    llm = ConversationalLM()
    vocab_size = llm.build_vocab(texts)
    print(f"✅ Vocabulaire : {vocab_size} tokens")
    
    # Optimiseur de séquences
    sequence_optimizer = SequenceOptimizer(
        min_length=8,
        optimal_length=60,  # Plus court pour efficacité
        max_length=100
    )
    
    # Préparer les données
    print("\n⚙️ Préparation des données...")
    train_size = int(0.9 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    train_batches = create_optimized_dataloader(train_texts, llm, sequence_optimizer, batch_size=6)
    val_batches = create_optimized_dataloader(val_texts, llm, sequence_optimizer, batch_size=6)
    
    print(f"✅ Train : {len(train_batches)} batches")
    print(f"✅ Val : {len(val_batches)} batches")
    
    # Modèle optimisé
    print("\n🧠 Initialisation du modèle optimisé...")
    model = OptimizedSuperNeuronConversational(
        vocab_size=vocab_size,
        embed_dim=96,  # Plus compact
        max_seq_length=100,
        num_branches=3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Paramètres : {total_params:,}")
    
    # Optimiseur et scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Gestionnaire d'entraînement
    trainer = TrainingOptimizer(model, device)
    
    # Entraînement
    print("\n🎯 Début de l'entraînement optimisé...")
    epochs = 25
    
    for epoch in range(epochs):
        print(f"\n📈 Époque {epoch+1}/{epochs}")
        
        # Entraînement
        train_metrics = trainer.train_epoch(train_batches, optimizer, scheduler)
        
        # Validation
        val_metrics = trainer.validate(val_batches)
        
        # Collecter métriques
        for key, value in train_metrics.items():
            trainer.metrics[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            trainer.metrics[f'val_{key}'].append(value)
        trainer.metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Affichage
        print(f"🔵 Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}, PPL: {train_metrics['perplexity']:.2f}")
        print(f"🔴 Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, PPL: {val_metrics['perplexity']:.2f}")
        
        # Early stopping
        if trainer.early_stopping_check(val_metrics['loss'], patience=8):
            print("⏹️ Early stopping déclenché !")
            break
        
        # Sauvegarde du meilleur modèle
        if val_metrics['loss'] == trainer.best_loss:
            trainer.save_checkpoint(epoch, optimizer, "best_optimized_model.pth")
    
    # Sauvegarde finale
    print("\n💾 Sauvegarde finale...")
    trainer.save_checkpoint(epoch, optimizer, "final_optimized_model.pth")
    trainer.plot_metrics("optimized_training_metrics.png")
    
    # Test de génération
    print("\n🎯 Test de génération optimisée...")
    model.eval()
    test_prompts = ["Bonjour", "Comment allez-vous", "Qu'est-ce que"]
    
    for prompt in test_prompts:
        generated = llm.generate(prompt, max_length=15, temperature=0.8)
        print(f"'{prompt}' → '{generated}'")
    
    print("\n🎉 Entraînement optimisé terminé !")

if __name__ == "__main__":
    main()
