import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.linalg import expm

# Define constants
H = np.array([[1, 10], [10, 2]])
totaltime = 5
deltat_real = 0.01
deltat_imag = 0.01
start = time.time()

# Initialize lists for data collection
t = []
alpha = []
beta = []
current_time = 0

# Initialize simulator
simulator = AerSimulator()

# Initial state preparation (|0⟩ → H|0⟩)
qc = QuantumCircuit(1)
qc.h(0)  # Apply Hadamard to create |+⟩ state

# Get initial state
current_state = Statevector.from_instruction(qc)
current_state = current_state.data

# Time evolution loop
while current_time <= totaltime:
    # Calculate imaginary time evolution operator
    U = expm(-H * deltat_imag)

    # Evolve the state
    evolved_state = np.dot(U, current_state)

    # Normalize
    current_state = evolved_state / np.linalg.norm(evolved_state)

    # Store probabilities
    alpha.append(np.real(current_state[0] * np.conjugate(current_state[0])))
    beta.append(np.real(current_state[1] * np.conjugate(current_state[1])))
    t.append(current_time)

    current_time += deltat_real

# Calculate ground state energy
final_state = current_state
evolved_final = np.dot(expm(-H * deltat_real), final_state)
E0 = (-1 / deltat_real) * np.log(
    np.linalg.norm(evolved_final) / np.linalg.norm(final_state)
)
print(f"Ground State Energy E0 = {E0:.6f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, alpha, label="|0⟩ probability")
plt.plot(t, beta, label="|1⟩ probability")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Quantum Imaginary Time Evolution")
plt.legend()
plt.grid(True)
plt.show()

elapsed = time.time() - start
print(f"Quantum algorithm completed in {elapsed:.2f} seconds")
# Print final state probabilities
print(f"\nFinal state probabilities:")
print(f"|0⟩: {alpha[-1]:.4f}")
print(f"|1⟩: {beta[-1]:.4f}")
