#!/usr/bin/env python3
"""
Optimiseur de séquences pour un apprentissage plus efficace
"""

import torch
import re
from typing import List, Tuple

class SequenceOptimizer:
    """Optimise la longueur des séquences pour l'entraînement"""
    
    def __init__(self, min_length=10, optimal_length=80, max_length=120):
        self.min_length = min_length
        self.optimal_length = optimal_length  
        self.max_length = max_length
        
    def smart_split(self, text: str, tokenizer) -> List[str]:
        """Découpe intelligent en préservant le sens"""
        
        # Tokenize le texte complet
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= self.optimal_length:
            return [text]  # Pas besoin de découper
        
        # Points de découpe naturels (ordre de priorité)
        split_patterns = [
            r'[.!?]\s+',     # Fin de phrases
            r'[,;]\s+',      # Virgules et point-virgules  
            r'\s+et\s+',     # Conjonctions
            r'\s+mais\s+',
            r'\s+donc\s+',
            r'\s+car\s+',
            r'\s+'           # Espaces (dernier recours)
        ]
        
        sequences = []
        current_text = text
        
        while current_text and len(tokenizer.encode(current_text)) > self.optimal_length:
            best_split = None
            best_score = 0
            
            # Chercher le meilleur point de découpe
            for pattern in split_patterns:
                matches = list(re.finditer(pattern, current_text))
                
                for match in matches:
                    split_pos = match.end()
                    prefix = current_text[:split_pos].strip()
                    
                    # Vérifier que le préfixe est dans la zone optimale
                    prefix_tokens = len(tokenizer.encode(prefix))
                    
                    if self.min_length <= prefix_tokens <= self.max_length:
                        # Score basé sur la proximité de la longueur optimale
                        score = 1.0 - abs(prefix_tokens - self.optimal_length) / self.optimal_length
                        
                        # Bonus pour les coupures naturelles (phrases)
                        if pattern == r'[.!?]\s+':
                            score *= 1.5
                        elif pattern in [r'[,;]\s+']:
                            score *= 1.2
                        
                        if score > best_score:
                            best_score = score
                            best_split = (prefix, current_text[split_pos:].strip())
            
            if best_split:
                prefix, remaining = best_split
                if prefix:  # Éviter les séquences vides
                    sequences.append(prefix)
                current_text = remaining
            else:
                # Forcer une coupure si aucune coupure naturelle
                target_tokens = min(self.optimal_length, len(tokenizer.encode(current_text)))
                
                # Estimer la position approximative
                char_per_token = len(current_text) / len(tokenizer.encode(current_text))
                approx_pos = int(target_tokens * char_per_token)
                
                # Chercher l'espace le plus proche
                space_pos = current_text.rfind(' ', 0, approx_pos)
                if space_pos > self.min_length * char_per_token:
                    sequences.append(current_text[:space_pos].strip())
                    current_text = current_text[space_pos:].strip()
                else:
                    break  # Impossible de découper proprement
        
        # Ajouter le reste s'il est assez long
        if current_text and len(tokenizer.encode(current_text)) >= self.min_length:
            sequences.append(current_text)
        
        return [seq for seq in sequences if seq.strip()]  # Filtrer les vides
    
    def create_overlapping_context(self, sequences: List[str], overlap_sentences=1) -> List[str]:
        """Crée un chevauchement entre séquences pour préserver le contexte"""
        if len(sequences) <= 1:
            return sequences
        
        overlapping_sequences = []
        
        for i, seq in enumerate(sequences):
            if i == 0:
                # Première séquence : pas de contexte précédent
                overlapping_sequences.append(seq)
            else:
                # Ajouter du contexte de la séquence précédente
                prev_sentences = sequences[i-1].split('.')
                
                # Prendre les dernières phrases comme contexte
                context_sentences = prev_sentences[-overlap_sentences:] if len(prev_sentences) > overlap_sentences else prev_sentences
                context = '. '.join(s.strip() for s in context_sentences if s.strip())
                
                if context and not context.endswith('.'):
                    context += '.'
                
                # Combiner contexte + séquence actuelle
                if context:
                    combined = f"{context} {seq}".strip()
                else:
                    combined = seq
                    
                overlapping_sequences.append(combined)
        
        return overlapping_sequences
    
    def optimize_dataset(self, texts: List[str], tokenizer) -> List[str]:
        """Optimise tout le dataset"""
        optimized_sequences = []
        
        for text in texts:
            # Découpage intelligent
            sequences = self.smart_split(text, tokenizer)
            
            # Ajouter chevauchement pour contexte
            overlapped = self.create_overlapping_context(sequences, overlap_sentences=1)
            
            optimized_sequences.extend(overlapped)
        
        return optimized_sequences
    
    def get_statistics(self, sequences: List[str], tokenizer) -> dict:
        """Statistiques sur les séquences optimisées"""
        lengths = [len(tokenizer.encode(seq)) for seq in sequences]
        
        return {
            'count': len(sequences),
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'optimal_ratio': sum(1 for l in lengths if l <= self.optimal_length * 1.2) / len(lengths) if lengths else 0
        }
