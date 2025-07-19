from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError
from qiskit.quantum_info import Statevector, Kraus
from qiskit.visualization import plot_histogram, plot_state_city, plot_bloch_multivector, circuit_drawer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ----------------------------
# TOGGLES
# ----------------------------
use_phonon_noise = True
plot_animated = True
compare_experimental_T1_T2 = True  # Hook for future real data

# ----------------------------
# PHYSICAL PARAMETERS
# ----------------------------
cz_gate_time = 0.01  # 10 ps
B_fields = {"No Magnetic Field": 0.1, "Strong Magnetic Field": 5.0}  # Tesla

def estimate_T1_T2_from_B(B_field_T):
    hbar = 1.055e-34
    gamma = 10 * 1.602e-19  # J
    rho = 2200  # kg/m^3
    c = 4300    # m/s
    delta = 1e10  # Hz
    delta_tilde = delta * np.exp(- (gamma**2 * (1e13)**2) / (2 * np.pi**2 * hbar * rho * c**5))
    Gamma_so = (gamma**2 * delta_tilde**3) / (4 * np.pi * hbar * rho * c**5)
    T1 = T2 = 1 / Gamma_so
    return T1, T2

# ----------------------------
# CIRCUIT AND NOISE MODEL
# ----------------------------
def create_circuit(use_cz=True):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cz(0, 1) if use_cz else qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def phonon_induced_kraus(T_phi, time):
    gamma = 1 / T_phi
    E0 = np.sqrt(np.exp(-gamma * time)) * np.eye(2)
    E1 = np.sqrt(1 - np.exp(-gamma * time)) * np.array([[1, 0], [0, 0]])
    return Kraus([E0, E1])

def build_noise_model(T1, T2, gate_time):
    noise_model = NoiseModel()
    if use_phonon_noise:
        kraus = phonon_induced_kraus(min(T1, T2), gate_time)
        noise_model.add_all_qubit_quantum_error(kraus, ['cz'])
    else:
        cz_error = thermal_relaxation_error(T1, T2, gate_time).tensor(
                    thermal_relaxation_error(T1, T2, gate_time))
        noise_model.add_all_qubit_quantum_error(cz_error, ['cz'])

    readout_error = ReadoutError([[0.99, 0.01], [0.01, 0.99]])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model

# ----------------------------
# SIMULATION + PLOTS
# ----------------------------
def simulate_and_plot(qc, noise_model, label):
    simulator = AerSimulator(method='density_matrix')
    transpiled_qc = transpile(qc, simulator)
    result = simulator.run(transpiled_qc, noise_model=noise_model, shots=1000).result()
    counts = result.get_counts()

    # Plot with matplotlib directly
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_histogram(counts, ax=ax)
    ax.set_title(f"Results: {label}", fontsize=16)
    ax.set_xlabel("Outcome", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()

    return counts

def show_statevector_visualizations():
    qc_sv = QuantumCircuit(2)
    qc_sv.h(0)
    qc_sv.cz(0, 1)
    sv = Statevector.from_instruction(qc_sv)
    plot_state_city(sv, title="Statevector - CZ Gate", figsize=(8, 6)).show()
    plot_bloch_multivector(sv, title="Bloch Vector", figsize=(8, 4)).show()

def save_circuit_images():
    cz = create_circuit(use_cz=True)
    cx = create_circuit(use_cz=False)
    circuit_drawer(cz, output='mpl', filename="cz_circuit.png")
    circuit_drawer(cx, output='mpl', filename="cnot_circuit.png")
    print("Circuit diagrams saved.")

def cz_truth_table():
    print("\n--- CZ Gate Truth Table ---")
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        qc = QuantumCircuit(2)
        if a: qc.x(0)
        if b: qc.x(1)
        qc.cz(0, 1)
        sv = Statevector.from_instruction(qc)
        print(f"|{a}{b}> → {sv.to_dict()}")

# ----------------------------
# DECOHERENCE PLOT
# ----------------------------
def plot_decoherence_curve(animated=False):
    times = np.linspace(0.1, 100, 200)
    g_values = [0.25, 0.5, 1.0, 2.0]
    colors = ['blue', 'black', 'green', 'red']
    fig, ax = plt.subplots(figsize=(12, 6))

    def purity(T1, T2, t):
        error = thermal_relaxation_error(T1, T2, t)
        kraus = Kraus(error)
        K0 = kraus.data[0]
        rho = K0 @ K0.conj().T
        return np.real(np.trace(rho @ rho))

    if animated:
        lines = [ax.plot([], [], label=f"g={g}", color=colors[i])[0] for i, g in enumerate(g_values)]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.set_title("Animated Decoherence (Purity vs Time)", fontsize=16)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Purity")
        ax.grid(True)
        ax.legend()

        def animate(i):
            for j, g in enumerate(g_values):
                T1 = T2 = 1e3 * g
                purities = [purity(T1, T2, t) for t in times[:i]]
                lines[j].set_data(times[:i], purities)
            return lines

        ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True)
        ani.save("animated_decoherence.gif", writer="pillow")
        print("Saved animated plot to animated_decoherence.gif")
        plt.show()

    else:
        for i, g in enumerate(g_values):
            T1 = T2 = 1e3 * g
            purities = [purity(T1, T2, t) for t in times]
            ax.plot(times, purities, label=f"g = {g}", color=colors[i], linewidth=2)

        ax.set_title("CZ Gate Decoherence vs Coupling Strength", fontsize=16)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Purity ρ₀₁(t)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig("decoherence_comparison_plot.png")
        print("Saved static decoherence plot to decoherence_comparison_plot.png")
        plt.show()

# ----------------------------
# MAIN EXECUTION
# ----------------------------
print("\n--- Simulating CZ Gate Under Magnetic Fields ---")
for label, B in B_fields.items():
    print(f"\n>>> {label} (B = {B} T)")
    T1, T2 = estimate_T1_T2_from_B(B)
    if compare_experimental_T1_T2:
        print(f"Estimated T1 = T2 = {T1:.2e} s")

    qc = create_circuit(use_cz=True)
    nm = build_noise_model(T1, T2, cz_gate_time)
    simulate_and_plot(qc, nm, label)

show_statevector_visualizations()
save_circuit_images()
cz_truth_table()
plot_decoherence_curve(animated=plot_animated)
