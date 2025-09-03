import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from super_neuron_stable import SuperNeuronLM, SimpleTokenizer


def load_trained_model(model_path='saved_models/super_neuron_model.pth', 
                      tokenizer_path='saved_models/tokenizer.pkl'):
    """Charge le modèle et tokenizer entraînés"""
    
    print("Chargement du modèle et tokenizer...")
    
    # Charger le tokenizer
    tokenizer = SimpleTokenizer.load_tokenizer(tokenizer_path)
    
    # Charger le modèle
    model = SuperNeuronLM.load_model(model_path)
    
    return model, tokenizer


def generate_text_inference(model, tokenizer, prompt, max_length=15, temperature=1.0):
    """Génère du texte avec le modèle chargé"""
    model.eval()
    
    token_ids = tokenizer.encode(prompt, max_length=24)
    input_tensor = torch.tensor([token_ids])
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor, use_routing=True)
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


def interactive_generation():
    """Mode interactif pour générer du texte"""
    print("=== SUPER NEURON - MODE INFÉRENCE ===")
    
    try:
        model, tokenizer = load_trained_model()
        print(f"Modèle chargé avec {tokenizer.vocab_size} mots dans le vocabulaire")
        print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
        
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Assurez-vous d'avoir d'abord entraîné le modèle avec super_neuron_stable.py")
        return
    
    print("\n" + "="*50)
    print("Tapez vos prompts (ou 'quit' pour quitter)")
    print("Exemples: 'Les algorithmes', 'L'intelligence artificielle'")
    print("="*50)
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Au revoir !")
                break
                
            if not prompt:
                continue
            
            # Génération avec différentes températures
            print(f"\n📝 Prompt: '{prompt}'")
            
            # Température basse (plus conservateur)
            result_low = generate_text_inference(model, tokenizer, prompt, 
                                               max_length=10, temperature=0.7)
            print(f"🎯 Temp 0.7: '{result_low}'")
            
            # Température normale
            result_mid = generate_text_inference(model, tokenizer, prompt, 
                                               max_length=10, temperature=1.0)
            print(f"🔄 Temp 1.0: '{result_mid}'")
            
            # Température haute (plus créatif)
            result_high = generate_text_inference(model, tokenizer, prompt, 
                                                max_length=10, temperature=1.3)
            print(f"🚀 Temp 1.3: '{result_high}'")
            
        except KeyboardInterrupt:
            print("\n\nAu revoir !")
            break
        except Exception as e:
            print(f"Erreur: {e}")


def batch_generation():
    """Test sur plusieurs prompts prédéfinis"""
    print("=== SUPER NEURON - TEST BATCH ===")
    
    try:
        model, tokenizer = load_trained_model()
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        return
    
    test_prompts = [
        "Les algorithmes",
        "L'intelligence artificielle",
        "Les réseaux de neurones",
        "Le traitement du langage",
        "La programmation",
        "Les données",
        "L'apprentissage automatique",
        "La cybersécurité",
        "Le cloud computing",
        "Les microservices"
    ]
    
    print(f"\nTest sur {len(test_prompts)} prompts:\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        result = generate_text_inference(model, tokenizer, prompt, max_length=12)
        print(f"{i:2d}. '{prompt}' → '{result}'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        batch_generation()
    else:
        interactive_generation()
