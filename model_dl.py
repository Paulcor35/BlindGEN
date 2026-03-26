from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

# Script à exécuter une seule fois pour sauvegarder le modèle physiquement
path = "gpt2_local"
if not os.path.exists(path):
    os.makedirs(path)

print("Téléchargement et sauvegarde locale...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.save_pretrained(path)
model.save_pretrained(path)
print(f"TERMINÉ : Modèle disponible en local dans '{path}'")
