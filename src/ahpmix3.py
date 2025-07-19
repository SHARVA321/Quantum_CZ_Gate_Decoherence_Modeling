import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==== PARAMETERS ====
omega = 1.0
V = 0.2
Omega = 0.2
tlist = np.linspace(0, 100, 1000)

# ==== Magnetic field parameters ====
B_ext = 5.0  # Tesla
g_factor = 2.0
mu_B = 5.788e-5  # eV/T
zeeman_split = g_factor * mu_B * B_ext
gamma_phi = 0.005 + 0.01 * B_ext

# ==== BASIS ====
q0 = basis(2, 0)
q1 = basis(2, 1)
basis_states = [tensor(q0, q0), tensor(q0, q1), tensor(q1, q0), tensor(q1, q1)]

# ==== Operators ====
I = qeye(2)
sm = destroy(2)
sp = sm.dag()
sz = sigmaz()

a1 = tensor(sm, I)
a2 = tensor(I, sm)
adag1 = a1.dag()
adag2 = a2.dag()
n1 = adag1 * a1
n2 = adag2 * a2
sz1 = tensor(sz, I)
sz2 = tensor(I, sz)

# ==== Initial state ====
psi0 = tensor(q1, q0)

def run_simulation(include_B=False):
    H0 = omega * (n1 + n2)
    Hint = V * n1 * n2
    Hl = Omega * (a2 + adag2)
    Hz = 0.5 * zeeman_split * (sz1 + sz2) if include_B else 0
    H = H0 + Hint + Hl + Hz

    c_ops = []
    if include_B:
        c_ops = [np.sqrt(gamma_phi) * sz1, np.sqrt(gamma_phi) * sz2]

    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[])
    
    probs = {label: [] for label in ['|00⟩', '|01⟩', '|10⟩', '|11⟩']}
    for state in result.states:
        probs['|00⟩'].append(np.abs(basis_states[0].overlap(state))**2)
        probs['|01⟩'].append(np.abs(basis_states[1].overlap(state))**2)
        probs['|10⟩'].append(np.abs(basis_states[2].overlap(state))**2)
        probs['|11⟩'].append(np.abs(basis_states[3].overlap(state))**2)
    
    purity = [state.purity() for state in result.states]
    fidelity_vals = [fidelity(psi0, state) for state in result.states]
    final_probs = [np.abs(state.overlap(result.states[-1]))**2 for state in basis_states]

    return probs, purity, fidelity_vals, final_probs

# ==== Run both simulations ====
probs_noB, purity_noB, fidelity_noB, final_probs_noB = run_simulation(include_B=False)
probs_B, purity_B, fidelity_B, final_probs_B = run_simulation(include_B=True)

# ==== PLOT 1: State Probabilities ====
plt.figure(figsize=(12, 6))
for label in probs_noB:
    plt.plot(tlist, probs_noB[label], '--', label=f'{label} (no B)')
    plt.plot(tlist, probs_B[label], '-', label=f'{label} (B={B_ext}T)')
plt.title("State Probabilities Over Time (CZ Gate)", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==== PLOT 2: Final State Probabilities ====
x = np.arange(4)
width = 0.35
labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, final_probs_noB, width, label='No B-field', color='skyblue')
plt.bar(x + width/2, final_probs_B, width, label=f'B = {B_ext} T', color='teal')
plt.xticks(x, labels)
plt.ylabel("Probability")
plt.title("Final State Probabilities Comparison")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==== PLOT 3: Purity ====
plt.figure(figsize=(8, 5))
plt.plot(tlist, purity_noB, '--', label="Purity (no B)", color='green')
plt.plot(tlist, purity_B, '-', label=f"Purity (B = {B_ext} T)", color='darkred')
plt.title("Purity Over Time")
plt.xlabel("Time")
plt.ylabel("Purity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==== PLOT 4: Fidelity ====
plt.figure(figsize=(8, 5))
plt.plot(tlist, fidelity_noB, '--', label="Fidelity (no B)", color='blue')
plt.plot(tlist, fidelity_B, '-', label=f"Fidelity (B = {B_ext} T)", color='orange')
plt.title("Fidelity to Initial State Over Time")
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
