import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from moai_paper_implementation import MOAIPaperCKKS

def plot_approximations():
    # Dossiers de sortie
    base_dir = "plots"
    for sub in ["gelu", "silu"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    degrees = [11, 15, 20, 22, 23, 24, 25, 29, 45]
    # On étend la zone pour bien voir l'explosion hors-clamping
    x_range = torch.linspace(-30, 30, 2000)
    
    # ─── FONCTIONS RÉELLES ───
    def real_gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
    
    def real_silu(x):
        return x * torch.sigmoid(x)

    # ─── GÉNÉRATION DES PLOTS GELU ───
    clamp_a, clamp_b = -10, 10
    y_real = real_gelu(x_range)


    for deg in degrees:
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, y_real, 'k-', lw=2, label='GELU Réel', alpha=0.5)
        plt.axvspan(clamp_a, clamp_b, color='gray', alpha=0.1, label=f'Zone Clamping [{clamp_a}, {clamp_b}]')
        
        coeffs = MOAIPaperCKKS.compute_gelu_minimax_poly_coeffs(degree=deg, a=clamp_a, b=clamp_b)
        
        # Simulation de l'évaluation avec clamping
        inner = math.sqrt(2/math.pi) * (x_range + 0.044715 * x_range**3)
        inner_clamped = torch.clamp(inner, clamp_a, clamp_b)
        
        c_rev = list(reversed(coeffs))
        p_res = torch.full_like(inner_clamped, c_rev[0])
        for i in range(1, len(c_rev)):
            p_res = p_res * inner_clamped + c_rev[i]
        y_approx = 0.5 * x_range * (1.0 + p_res)
        
        # Tracé de la version SANS clamping pour voir l'explosion (instabilité)
        p_res_unclamped = torch.full_like(inner, c_rev[0])
        for i in range(1, len(c_rev)):
            p_res_unclamped = p_res_unclamped * inner + c_rev[i]
        y_approx_unclamped = 0.5 * x_range * (1.0 + p_res_unclamped)

        plt.plot(x_range, y_approx, 'b-', label=f'MOAI (Clamped) Deg {deg}')
        plt.plot(x_range, y_approx_unclamped, 'r--', alpha=0.3, label=f'Instabilité (Unclamped)')
        
        plt.ylim(-5, 20)
        plt.title(f'GELU : Impact du Degré {deg} et du Clamping', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/gelu/gelu_deg_{deg}.png')
        plt.close()

    # ─── GÉNÉRATION DES PLOTS SILU ───
    clamp_a_s, clamp_b_s = -10, 10
    y_real_s = real_silu(x_range)

    for deg in degrees:
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, y_real_s, 'k-', lw=2, label='SiLU Réel', alpha=0.5)
        plt.axvspan(clamp_a_s, clamp_b_s, color='gray', alpha=0.1, label=f'Zone Clamping [{clamp_a_s}, {clamp_b_s}]')
        
        coeffs = MOAIPaperCKKS.compute_silu_poly_coeffs(degree=deg, a=clamp_a_s, b=clamp_b_s)
        
        x_clamped = torch.clamp(x_range, clamp_a_s, clamp_b_s)
        c_rev = list(reversed(coeffs))
        
        p_res = torch.full_like(x_clamped, c_rev[0])
        for i in range(1, len(c_rev)):
            p_res = p_res * x_clamped + c_rev[i]
        
        p_res_unclamped = torch.full_like(x_range, c_rev[0])
        for i in range(1, len(c_rev)):
            p_res_unclamped = p_res_unclamped * x_range + c_rev[i]

        plt.plot(x_range, p_res, 'g-', label=f'MOAI (Clamped) Deg {deg}')
        plt.plot(x_range, p_res_unclamped, 'r--', alpha=0.3, label=f'Instabilité (Unclamped)')
        
        plt.ylim(-10, 25)
        plt.title(f'SiLU : Impact du Degré {deg} et du Clamping', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plots/silu/silu_deg_{deg}.png')
        plt.close()

    print("Tous les graphiques ont été générés dans le dossier 'plots/'.")


if __name__ == "__main__":
    plot_approximations()
