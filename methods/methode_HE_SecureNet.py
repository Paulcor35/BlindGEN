import time
import random

def encrypt(text):
    time.sleep(0.1)  # Rapide
    payload = f"[AES-256] 0x{random.randint(100, 999)}_" + "".join(random.choices("ABCDEF", k=20))
    return payload, 0.1 # Retourne le texte chiffré et le temps d'exécution