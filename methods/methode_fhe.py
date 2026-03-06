import time
import random

def encrypt(text):
    time.sleep(1.2)  # Plus lent (simulation de chiffrement homomorphe)
    payload = f"[FHE-LATTICE] 0x{random.randint(1000, 9999)}_" + "".join(random.choices("0123456789", k=50))
    return payload, 1.2