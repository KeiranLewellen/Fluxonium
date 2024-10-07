import numpy as np
import math
from quspin.operators import hamiltonian, exp_op
from quspin.basis import boson_basis_1d
from quspin.tools.Floquet import Floquet, Floquet_t_vec
from quspin.tools.evolution import ED_state_vs_time
from quspin.tools.measurements import obs_vs_time
import matplotlib.pyplot as plt
import h5py

# Imports the fluxonium Hamiltonians file for use here
import Hamiltonians_fluxonium_ED_chain_cosine_drive as H


# Functions that evolve systems

def evolve_with_cos_drive(n, L, cycle_num, EC, EL, EJ, A, omega, start_state, obvs_H, cpu_num):
    """This is the main function that defines the Hamiltonians, efficiently evolves the wave function,
    and calculates the expectation of obvs_H and the fidelity with the initial state"""

    # Initializes parameters
    k = n
    n_coeff = (1j / 2) * (EL / (2 * EC)) ** (1 / 4)
    phi_coeff = ((2 * EC) / EL) ** (1 / 4)
    basis = boson_basis_1d(L, sps=n)  # particle non conserving basis, n states per site

    # Hamiltonian parts
    H_EC = (4 * EC) * H.n2(L, n_coeff, basis)
    H_EL = (EL / 2) * H.phi2(L, phi_coeff, basis)
    H_EJ_driven = -EJ * H.driven_cos_ordered(k, L, phi_coeff, omega, A, basis)
    H_V = V * H.nn(L, n_coeff, basis)
    H_first_order_magnus = H.magnus_residual(k, L, EC, EJ, n_coeff, phi_coeff, omega, A, basis)

    # Total Hamiltonians
    H_tot_driven = H_EC + H_EL + H_EJ_driven + H_V # Time dependent driven Hamiltonian (in the rotating basis)
    H_magnus = H_EC + H_EL + H_first_order_magnus + H_V # Time independent Hamiltonian up to first order in the Magnus expansion

    # Initialization
    energies, states = H_magnus.eigh()  # eigenvalue and eigenstate of H_magnus. Change for other system initializations
    psi0 = states[:, start_state]  # start_state controls which excited state the system is initialized in
    obvs_0 = obvs_H.expt_value(psi0)
    print("obvs_0: ", obvs_0)

    # Evolves system using the known Floquet eigenvectors
    # The evolution time is independent of the number of cycles considered (though the accuracy will still decrease)
    print("diagonalizing system")
    t_vec = Floquet_t_vec(omega, cycle_num, len_T=1)
    floq = Floquet({'H': H_tot_driven, 'T': t_vec.T}, VF=True, n_jobs=cpu_num) # Do we need to orthogonalize the eigenstates here?
    floq_states = floq.VF
    floq_energies = floq.EF
    print("evolving system")
    psi1_time = ED_state_vs_time(psi0, floq_energies, floq_states, t_vec.values, iterate=True)

    # Calculates the observables of interest
    obvs_t = []
    fid_t = []
    for psi_t in psi1_time:
        expectation_val = obvs_H.expt_value(psi_t)
        fidelity_val = np.linalg.norm(np.vdot(psi0, psi_t)) ** 2
        obvs_t.append(expectation_val)
        fid_t.append(fidelity_val)

    output = {"obvs_expectation": np.array(obvs_t) / obvs_0, "fidelity": np.array(fid_t), "t_indices": t_vec.indices}
    return output


# Coefficients

EC = 0.33
EL = 1
EJ = 12.58
V = 0.05
omega = 15
A = 2.4048*omega

L = 2
n = 10
k = n

cycle_num = 50
start_state = 1
site_i = 0

cpu_num = 4

# System evolution

n_coeff = (1j / 2) * (EL / (2 * EC)) ** (1 / 4)
phi_coeff = ((2 * EC) / EL) ** (1 / 4)
basis = boson_basis_1d(L, sps=n)
obvs_H = H.n_i(site_i, n_coeff, basis)

output = evolve_with_cos_drive(n, L, cycle_num, EC, EL, EJ, A, omega, start_state, obvs_H, cpu_num)

print("obvs: ", output['obvs_expectation'])
print("fidelity: ", output['fidelity'])

