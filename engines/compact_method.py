import numpy as np
from numpy.polynomial import Chebyshev as T
import torch
import torch.nn.functional as F

class CompactActivation:
    """
    Implements the 'Compact' method for approximating complex activation functions
    using piece-wise polynomials, optimized for MPC and FHE.
    Reference: Mazharul Islam et al., "Compact: Approximating Complex Activation Functions for Secure Computation" (PoPETS 2024)
    """

    def __init__(self, func_type='silu', m=20, k=3, range_val=(-5, 5)):
        """
        Args:
            func_type (str): 'silu', 'gelu', or 'mish'
            m (int): Number of piece-wise polynomial segments
            k (int): Degree of each polynomial segment
            range_val (tuple): The interval [a, b] to approximate over. Outside this, 
                              asymptotic behavior is used (usually 0 or x).
        """
        self.func_type = func_type.lower()
        self.m = m
        self.k = k
        self.start, self.end = range_val
        self.pieces = []
        self._generate_pieces()

    def _target_function(self, x):
        if self.func_type == 'silu':
            # SiLU(x) = x / (1 + exp(-x))
            return x / (1 + np.exp(-x))
        elif self.func_type == 'gelu':
            # GELU(x) = x * P(X <= x)
            # Using the fast approximation formula from the paper/Transformers
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif self.func_type == 'mish':
            # Mish(x) = x * tanh(ln(1 + exp(x)))
            return x * np.tanh(np.log(1 + np.exp(x)))
        else:
            raise ValueError(f"Unsupported activation: {self.func_type}")

    def _generate_pieces(self):
        """
        Generates m piece-wise polynomials using Chebyshev interpolation over the range.
        Note: This is a simplified version of the full 'Compact' search algorithm.
        """
        x_points = np.linspace(self.start, self.end, self.m + 1)
        
        for i in range(self.m):
            a, b = x_points[i], x_points[i+1]
            
            # Use Chebyshev nodes for interpolation within [a, b]
            # Map [-1, 1] to [a, b]
            def get_y(t):
                return self._target_function(t)
            
            # Chebyshev interpolation of degree k
            # Using numpy.polynomial.Chebyshev.fit
            # This is robust and minimizes the maximum error (minimax)
            poly = T.fit(np.linspace(a, b, self.k + 5), 
                         get_y(np.linspace(a, b, self.k + 5)), 
                         deg=self.k, domain=[a, b])
            
            # Store the polynomial and its boundaries
            self.pieces.append({
                'range': (a, b),
                'poly': poly
            })

    def eval_plain(self, x):
        """
        Evaluation in plaintext (numpy).
        Handle asymptotic behavior first.
        """
        x_scalar = np.array(x)
        
        # Asymptotics
        if x_scalar < self.start:
            return 0.0
        if x_scalar > self.end:
            return float(x_scalar)
        
        # Find the correct piece
        for piece in self.pieces:
            a, b = piece['range']
            if a <= x_scalar <= b:
                return piece['poly'](x_scalar)
        
        return 0.0 # Fallback

    def eval_torch(self, x_tensor):
        """
        Evaluation using PyTorch tensors (for integration with POC).
        Uses masks to handle intervals efficiently.
        """
        # Ensure we are working with torch
        if not isinstance(x_tensor, torch.Tensor):
            x_tensor = torch.tensor(x_tensor, dtype=torch.float32)
            
        res = torch.zeros_like(x_tensor)
        
        # Asymptotics
        res[x_tensor < self.start] = 0.0
        res[x_tensor > self.end] = x_tensor[x_tensor > self.end]
        
        # Range pieces
        for piece in self.pieces:
            a, b = piece['range']
            mask = (x_tensor >= a) & (x_tensor <= b)
            if mask.any():
                # Convert Chebyshev to normal coefficients for evaluation
                # Using normal polynomial form: c0 + c1*x + c2*x^2 + ...
                poly = piece['poly']
                coeffs = poly.convert(kind=np.polynomial.Polynomial).coef
                
                # Evaluate: c0 + c1*x + c2*x^2 + ...
                piece_res = torch.zeros_like(x_tensor[mask])
                for idx, c in enumerate(coeffs):
                    piece_res += c * (x_tensor[mask] ** idx)
                res[mask] = piece_res
                
        return res

    def get_piece_info(self, index):
        """Returns bounds and coefficients for a specific piece (index or value range)."""
        piece = self.pieces[index]
        a, b = piece['range']
        coeffs = piece['poly'].convert(kind=np.polynomial.Polynomial).coef
        return {
            'range': (a, b),
            'coeffs': coeffs.tolist()
        }

    def export_all_coeffs(self):
        """Exports all piece coefficients as a list of dicts for deployment."""
        return [self.get_piece_info(i) for i in range(self.m)]

# --- Section pour tester l'approche crude mentionnée dans le papier ---

def silu_crude(x):
    """
    Equation 7 approx (paper): F_crd_silu(x) = x * HardSigmoid(x)
    Simple version: x * torch.clamp(x + 0.5, 0.0, 1.0)
    """
    return x * torch.clamp(x + 0.5, 0.0, 1.0)

def gelu_crude(x):
    """
    Proposed crude approximation for GELU based on hard-sigmoid
    """
    # GELU(x) ~ x * sigmoid(1.702 * x)
    return x * torch.clamp(1.702 * x + 0.5, 0.0, 1.0)

def compare_methods(func_type='silu'):
    print(f"\n--- Comparison for {func_type.upper()} ---")
    x = torch.linspace(-8, 8, 1000)
    
    if func_type == 'silu':
        y_true = F.silu(x)
        y_crude = silu_crude(x)
    else:
        # torch.nn.functional.gelu is only available on tensors
        y_true = F.gelu(x)
        y_crude = gelu_crude(x)
        
    # Compact configurations
    configs = [
        (4, 2), # 4 pieces, deg 2 (total 12 coeffs)
        (10, 3), # 10 pieces, deg 3 (total 40 coeffs)
        (20, 2), # 20 pieces, deg 2 (total 60 coeffs)
    ]
    
    for m, k in configs:
        compact = CompactActivation(func_type=func_type, m=m, k=k)
        y_comp = compact.eval_torch(x)
        mse = torch.mean((y_true - y_comp)**2)
        print(f"Compact (m={m}, k={k}): MSE = {mse:.2e}")
        
    mse_crude = torch.mean((y_true - y_crude)**2)
    print(f"Crude Approx: MSE = {mse_crude:.2e}")

    # Single high-degree polynomial (often best for FHE if range is small)
    poly_single = T.fit(x[(x >= -5) & (x <= 5)].numpy(), 
                        y_true[(x >= -5) & (x <= 5)].numpy(), deg=6, domain=[-5, 5])
    y_single = torch.zeros_like(x)
    y_single[x < -5] = 0
    y_single[x > 5] = x[x > 5]
    mask = (x >= -5) & (x <= 5)
    coeffs_single = poly_single.convert(kind=np.polynomial.Polynomial).coef
    piece_res = torch.zeros_like(x[mask])
    for idx, c in enumerate(coeffs_single):
        piece_res += c * (x[mask] ** idx)
    y_single[mask] = piece_res
    
    mse_single = torch.mean((y_true - y_single)**2)
    print(f"Single Deg 6 Poly: MSE = {mse_single:.2e}")

if __name__ == "__main__":
    compare_methods('silu')
    compare_methods('gelu')
