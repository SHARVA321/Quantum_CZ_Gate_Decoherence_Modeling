# Decoherence Modeling of a Controlled-Z Quantum Gate under Strong Magnetic Fields

## Abstract/Overview
This project investigates the decoherence behavior of a Controlled-Z (CZ) quantum gate in semiconductor quantum dots, particularly under the influence of strong magnetic fields. It leverages spin-boson and Zeeman dynamics to model the interactions and evaluate the gate's coherence properties under realistic conditions.

## Key Features/Contributions
* **Quantum Gate Simulation:** Simulated the evolution of a Controlled-Z (CZ) gate using both QuTiP (for Hamiltonian-driven dynamics) and Qiskit (for gate-level abstraction).
* **Decoherence Analysis:** Performed a comparative analysis of key metrics, including state probabilities, purity, and fidelity, in the presence and absence of a 5 Tesla magnetic field.
* **Magnetic Field Impact:** Demonstrated that strong magnetic fields accelerate decoherence by introducing significant dephasing, leading to a notable decline in quantum state purity and fidelity over time.
* **Theoretical Modeling:** Incorporated theoretical models to describe phonon-induced decoherence, charge noise suppression, and spin relaxation mechanisms within the quantum system.

## Methodology
The study employed a simulation-based theoretical approach:
* **QuTiP:** Used for master equation evolution to model the physical decoherence, including magnetic-field-dependent Zeeman splitting and collapse operators tied to phase damping.
* **Qiskit:** Utilized for gate-level abstraction, incorporating Kraus and thermal noise models to analyze logic fidelity.
* **Parameters:** Simulations were run with parameters such as $\omega=1.0$, $V=0.2$, $\Omega=0.2$, and a magnetic field $B=5~T$.

## Results
The simulations revealed that the inclusion of a strong magnetic field (5T) significantly increases decoherence, leading to a sharp decline in both purity and fidelity over time compared to the no-magnetic-field scenario. This indicates a substantial loss of quantum coherence and highlights the detrimental effect of such fields on qubit integrity in CZ gate operations under the simulated conditions.

## Visualizations
The `plots/` directory contains detailed graphs illustrating:
* State Probabilities Over Time (CZ Gate)
* Purity Over Time
* Fidelity to Initial State Over Time

Example:
![State Probabilities Over Time](plots/state_probabilities_over_time.png)

## Technologies Used
* `Python`
* `QuTiP`
* `Qiskit`
* `NumPy`
* `Matplotlib`

## Installation/Usage
To run the simulation scripts:
1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/Quantum-CZ-Gate-Decoherence-Modeling.git](https://github.com/YourUsername/Quantum-CZ-Gate-Decoherence-Modeling.git)
    cd Quantum-CZ-Gate-Decoherence-Modeling
    ```
2.  Install the required Python libraries:
    ```bash
    pip install qutip qiskit numpy matplotlib
    ```
3.  Navigate to the `src/` directory and run the simulation scripts:
    ```bash
    python src/simple_version_script.py
    python src/ahpmix3_simulation.py
    ```

## Full Report
For a comprehensive understanding of the project, including detailed theoretical underpinnings, design, and analysis, please refer to the full report:
[Decoherence Modeling of a CZ Gate under Strong Magnetic Fields Report](docs/Decoherence_Modeling_CZ_Gate_Report.pdf)

## Contributors
* Your Name (PES1UG22EC269)
* Aprameya Kulkarni (PES1UG22EC914)
