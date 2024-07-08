from quspin.operators import hamiltonian, exp_op   # Hamiltonians and operators# operators
from quspin.basis import boson_basis_1d  # Hilbert space boson basis
import numpy as np  # generic math functions
import math
from quspin.tools.measurements import obs_vs_time
from quspin.tools.Floquet import Floquet, Floquet_t_vec
import matplotlib.pyplot as plt
from scipy.linalg import cosm, inv
import h5py


def sin(t, omega):
    return np.sin(omega*t)


def cos_omega(t, omega):
    return (-1/omega)*np.cos(omega*t)


def square_drive(t, omega):  # Defines a square drive used to drive n
    return np.sign(np.sin(omega * t))


def triangular_drive(t, omega):  # Defines the integral of the square drive -- would be used to drive phi in the static frame
    return (2 * np.pi / omega) * np.abs(t * omega / (2 * np.pi) - np.floor(t * omega / (2 * np.pi) + 1/2))


def cos_triangular_drive(t, omega, A):  # Defines the cosine of the triangular drive
    return np.cos(A*((2 * np.pi / omega) * np.abs(t * omega / (2 * np.pi) - np.floor(t * omega / (2 * np.pi) + 1/2))))


def sin_triangular_drive(t, omega, A):  # Defines the sine of the triangular drive
    return np.sin(A*((2 * np.pi / omega) * np.abs(t * omega / (2 * np.pi) - np.floor(t * omega / (2 * np.pi) + 1/2))))


def make_ordered_exp_b(k, pm_str, phi_coeff, basis):
    '''caculates e^(+-i*phi_coeff*b^+), pm_str = "+", "-" defines whether it is a plus or minus'''
    static_exp_phi = []
    for i in range(k+1):
        j = i
        pm_j = pm_str*j
        interactj = [0] * j
        interactj = [[(1/math.factorial(j))*((1j*phi_coeff)**j)] + interactj]
        static_exp_phi.append([pm_j, interactj])
    H_exp_b = hamiltonian(static_exp_phi, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_exp_b


def make_ordered_exp_b_cos_t(k, pm_str, phi_coeff, omega, A, basis):
    dynamic_exp_phi = []
    for i in range(k+1):
        j = i
        pm_j = pm_str*j
        interactj = [0] * j
        interactj = [[(1/math.factorial(j))*((1j*phi_coeff)**j)] + interactj]
        dynamic_exp_phi.append([pm_j, interactj, cos_triangular_drive, [omega, A]])
    H_exp_b = hamiltonian([], dynamic_exp_phi, dtype=np.complex64, basis=basis, check_herm=False)
    return H_exp_b


def make_ordered_exp_b_sin_t(k, pm_str, phi_coeff, omega, A, basis):
    dynamic_exp_phi = []
    for i in range(k+1):
        j = i
        pm_j = pm_str*j
        interactj = [0] * j
        interactj = [[(1/math.factorial(j))*((1j*phi_coeff)**j)] + interactj]
        dynamic_exp_phi.append([pm_j, interactj, sin_triangular_drive, [omega, A]])
    H_exp_b = hamiltonian([], dynamic_exp_phi, dtype=np.complex64, basis=basis, check_herm=False)
    return H_exp_b


def cos_cos_t_ordered_Hamiltonian(k, phi_coeff, omega, A, basis):
    H_exp_bp_cos_t = make_ordered_exp_b_cos_t(k, '+', phi_coeff, omega, A, basis)
    H_exp_bm = make_ordered_exp_b(k, '-', phi_coeff, basis)
    H_exp_mbp_cos_t = make_ordered_exp_b_cos_t(k, '+', -1*phi_coeff, omega, A, basis)
    H_exp_mbm = make_ordered_exp_b(k, '-', -1*phi_coeff, basis)
    non_comuting_coeff = np.exp(-1*(phi_coeff**2)/2)/2
    ### Note that the next line calculates cos by writing it in terms of exponentials then using BCH to normal order them
    ### The normal ordering is done before we truncate the Hilbert space so the order of operations is correct
    ### In each term, one of the exponetials is a dynamic Hamiltonian which is multiplied by appropriate cosine drive
    cos_cos_t_ordered_H = non_comuting_coeff*(H_exp_bp_cos_t*H_exp_bm + H_exp_mbp_cos_t*H_exp_mbm)
    return cos_cos_t_ordered_H


def sin_sin_t_ordered_Hamiltonian(k, phi_coeff, omega, A, basis):  # Check over
    H_exp_bp_sin_t = make_ordered_exp_b_sin_t(k, '+', phi_coeff, omega, A, basis)
    H_exp_bm = make_ordered_exp_b(k, '-', phi_coeff, basis)
    H_exp_mbp_sin_t = make_ordered_exp_b_sin_t(k, '+', -1*phi_coeff, omega, A, basis)
    H_exp_mbm = make_ordered_exp_b(k, '-', -1*phi_coeff, basis)
    non_comuting_coeff = np.exp(-1*(phi_coeff**2)/2)/(2*1j)
    sin_sin_t_ordered_H = non_comuting_coeff*(H_exp_bp_sin_t*H_exp_bm - H_exp_mbp_sin_t*H_exp_mbm)
    return sin_sin_t_ordered_H


def evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state):
    '''This is the main function that defines the Hamiltonians, evolves the wave function, and calculates E_fid and fidelity'''
    n_coeff = (1j / 2) * (EL / (2 * EC)) ** (1 / 4)
    phi_coeff = ((2 * EC) / EL) ** (1 / 4)

    basis = boson_basis_1d(1, sps=n)  # particle non conserving basis, n states per site

    # Note that for these next two terms, we normal order the +,- (with proper commutator) before truncating them
    static_n2 = [["++", [[n_coeff ** 2, 0, 0]]], ["--", [[n_coeff ** 2, 0, 0]]], ["+-", [[-1 * (n_coeff ** 2), 0, 0]]],
                 ["+-", [[-1 * (n_coeff ** 2), 0, 0]]]]
    static_phi2 = [["++", [[phi_coeff ** 2, 0, 0]]], ["--", [[phi_coeff ** 2, 0, 0]]], ["+-", [[phi_coeff ** 2, 0, 0]]],
                   ["+-", [[phi_coeff ** 2, 0, 0]]]]

    # Hamiltonian parts
    H_EC = (4 * EC) * hamiltonian(static_n2, [], dtype=np.complex64, basis=basis)
    H_EL = (EL / 2) * hamiltonian(static_phi2, [], dtype=np.complex64, basis=basis)
    H_EJ_driven = -EJ*(cos_cos_t_ordered_Hamiltonian(k, phi_coeff, omega, A, basis) + sin_sin_t_ordered_Hamiltonian(k, phi_coeff, omega, A, basis))

    # static and dynamic hamiltonians
    H_static = H_EC + H_EL
    H_tot = H_EC + H_EL + H_EJ_driven

    # Initialization
    E2, V2 = H_static.eigh()  # eigenvalue and eigenstate of H_bosonic
    psi0 = V2[:, start_state]  # initialization, k means kth excited state
    E_EL_0 = H_static.expt_value(psi0)
    print("E0: ", E_EL_0)

    print("evolving system")

    time_list = Floquet_t_vec(omega, cycle_num, len_T=1).vals
    t_indices = Floquet_t_vec(omega, cycle_num, len_T=1).strobo.inds
    psi_t_gen = H_tot.evolve(psi0, 0, time_list, iterate=True)

    obvs_n = obs_vs_time(psi_t_gen, time_list, dict(E=H_static), return_state=True)

    E_EL_T = obvs_n["E"]
    psi_t = obvs_n["psi_t"]
    overlap_list = [np.linalg.norm(np.vdot(psi0, psi_t[:, i]))**2 for i in range(psi_t.shape[1])]
    output = {"E_fid": E_EL_T / E_EL_0, "Fidelity": overlap_list, "t_indices": t_indices}
    return output


#### Energy fidelity graphing functions

def E_fid_vs_alpha_graph(n, k, EC, EL, EJ, omega, start_state, cycle_num, alpha_list):
    ratio_list = np.linspace(alpha_list[0], alpha_list[1], alpha_list[2])
    end_fid_list = []
    for alpha in ratio_list:
        A = alpha * omega
        output = evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
        E_fid = output["E_fid"]
        end_fid_list.append(E_fid[-1])
    print("end_fid_list: ", end_fid_list)

    plt.plot(ratio_list, end_fid_list)
    plt.plot(ratio_list, np.ones(len(ratio_list)), linestyle=':')
    plt.title("Single Fluxonium, E fidelity vs alpha", fontsize=15)
    plt.ylabel("E fid after "+str(cycle_num)+" cycles", fontsize=15)
    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.grid()
    plt.savefig("Plots/Fluxonium_E_vs_alpha_simplified_square_wave_n="+str(n)+"_(EC,EL,EJ)="+str([EC, EL, EJ])+"_omega="
                + str(omega)+"_alpha="+str(alpha_list)+"_start_state="+str(start_state)+"_cycle_num="
                + str(cycle_num)+".pdf")


def E_avg_fid_vs_alpha_graph(n, k, EC, EL, EJ, omega, start_state, cycle_num, alpha_list):
    ratio_list = np.linspace(alpha_list[0], alpha_list[1], alpha_list[2])
    end_fid_list = []
    for alpha in ratio_list:
        A = alpha * omega
        output = evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
        E_fid = output["E_fid"]
        end_fid_list.append(np.sum(E_fid)/len(E_fid))
    print("end_fid_list: ", end_fid_list)

    plt.plot(ratio_list, end_fid_list)
    plt.plot(ratio_list, np.ones(len(ratio_list)), linestyle=':')
    plt.title("Single Fluxonium, E fidelity vs alpha", fontsize=15)
    plt.ylabel("E fid after "+str(cycle_num)+" cycles", fontsize=15)
    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.grid()
    plt.savefig("Plots/Fluxonium_E_avg_vs_alpha_simplified_square_wave_n="+str(n)+"_(EC,EL,EJ)="+str([EC, EL, EJ])+"_omega="
                + str(omega)+"_alpha="+str(alpha_list)+"_start_state="+str(start_state)+"_cycle_num="
                + str(cycle_num)+".pdf")


def E_fid_vs_omega_graph(n, k, EC, EL, EJ, omega_list, start_state, cycle_num, alpha):
    omega_arr = np.linspace(omega_list[0], omega_list[1], omega_list[2])
    end_fid_list = []
    for omega in omega_arr:
        print("omega: ", omega)
        A = alpha * omega
        output = evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
        E_fid = output["E_fid"]
        end_fid_list.append(np.sum(E_fid)/len(E_fid))
    print("end_fid_list: ", end_fid_list)

    plt.plot(omega_arr, end_fid_list)
    plt.plot(omega_arr, np.ones(len(omega_arr)), linestyle=':')
    plt.title(r"Single Fluxonium, E fidelity vs omega, $\alpha$="+str(alpha), fontsize=15)
    plt.ylabel("average E fid over "+str(cycle_num)+" cycles", fontsize=15)
    plt.xlabel(r"$\omega$", fontsize=15)
    plt.grid()
    plt.savefig("Plots/Fluxonium_E_avg_vs_omega_simplified_square_wave_n="+str(n)+"_(EC,EL,EJ)="+str([EC, EL, EJ])+"_omega="
                + str(omega_list)+"_alpha="+str(alpha)+"_start_state="+str(start_state)+"_cycle_num="
                + str(cycle_num)+".pdf")


def E_avg_fid_vs_alpha_and_omega_heat_map(n, k, EC, EL, EJ, omega_list, start_state, cycle_num, alpha_list): # I made this in a rush and the x and y tick labels need to be changed to be correct 
    ratio_list = np.linspace(alpha_list[0], alpha_list[1], alpha_list[2])
    omega_arr = np.linspace(omega_list[0], omega_list[1], omega_list[2])

    end_fid_list_list = []
    for alpha in ratio_list:
        end_fid_alpha_list = []
        for omega in omega_arr:
            print("(alpha, omega) ", (alpha,omega))
            A = alpha * omega
            output = evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
            E_fid = output["E_fid"]
            end_fid_alpha_list.append(np.sum(E_fid)/len(E_fid))
        end_fid_list_list.append(end_fid_alpha_list)
    print("end_fid_list: ", end_fid_list_list)
    end_fid_arr = np.real(np.array(end_fid_list_list))

    plt.imshow(end_fid_arr)
    plt.colorbar()

    h5f = h5py.File("Plots/Fluxonium_E_avg_vs_alpha_and_omega_simplified_square_wave_n=" + str(n) + "_(EC,EL,EJ)=" + str(
        [EC, EL, EJ]) + "_omega="
                + str(omega_list) + "_alpha=" + str(alpha_list) + "_start_state=" + str(start_state) + "_cycle_num="
                + str(cycle_num) + ".h5", 'w')
    h5f.create_dataset('data_1', data=end_fid_arr)

    plt.title(r"Single Fluxonium, E fidelity vs $\alpha$ and $\omega$", fontsize=15)
    plt.ylabel(r"$\alpha$", fontsize=15)
    plt.xlabel(r"$\omega$", fontsize=15)

    plt.savefig("Plots/Fluxonium_E_avg_vs_alpha_and_omega_simplified_square_wave_n=" + str(n) + "_(EC,EL,EJ)=" + str(
        [EC, EL, EJ]) + "_omega="
                + str(omega_list) + "_alpha=" + str(alpha_list) + "_start_state=" + str(start_state) + "_cycle_num="
                + str(cycle_num) + ".pdf")



#### Fidelity graphing functions

def fidelity_avg_vs_alpha_graph(n, k, EC, EL, EJ, omega, start_state, cycle_num, alpha_list):
    ratio_list = np.linspace(alpha_list[0], alpha_list[1], alpha_list[2])
    end_fid_list = []
    for alpha in ratio_list:
        A = alpha * omega
        output = evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
        fid_list = output["Fidelity"]
        end_fid_list.append(np.sum(fid_list)/len(fid_list))
    print("end_fid_list: ", end_fid_list)

    plt.plot(ratio_list, end_fid_list)
    plt.plot(ratio_list, np.ones(len(ratio_list)), linestyle=':')
    plt.title("Single Fluxonium, fidelity vs alpha", fontsize=15)
    plt.ylabel("Fidelity after "+str(cycle_num)+" cycles", fontsize=15)
    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.grid()
    plt.savefig("Plots/Fluxonium_fidelity_avg_vs_alpha_simplified_square_wave_n="+str(n)+"_(EC,EL,EJ)="+str([EC, EL, EJ])+"_omega="
                + str(omega)+"_alpha="+str(alpha_list)+"_start_state="+str(start_state)+"_cycle_num="
                + str(cycle_num)+".pdf")


def fid_avg_vs_alpha_and_omega_heat_map(n, k, EC, EL, EJ, omega_list, start_state, cycle_num, alpha_list): # I made this in a rush and the x and y tick labels need to be changed to be correct 
    ratio_list = np.linspace(alpha_list[0], alpha_list[1], alpha_list[2])
    omega_arr = np.linspace(omega_list[0], omega_list[1], omega_list[2])

    end_fid_list_list = []
    for alpha in ratio_list:
        end_fid_alpha_list = []
        for omega in omega_arr:
            print("(alpha, omega) ", (alpha, omega))
            A = alpha * omega
            output = evolve_with_square_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
            fid_list = output["Fidelity"]
            end_fid_alpha_list.append(np.sum(fid_list)/len(fid_list))
        end_fid_list_list.append(end_fid_alpha_list)
    print("end_fid_list: ", end_fid_list_list)
    end_fid_arr = np.real(np.array(end_fid_list_list))

    plt.imshow(end_fid_arr)
    plt.colorbar()

    plt.title(r"Single Fluxonium, fidelity avg vs $\alpha$ and $\omega$", fontsize=15)
    plt.ylabel(r"$\alpha$", fontsize=15)
    plt.xlabel(r"$\omega$", fontsize=15)

    plt.savefig("Plots/Fluxonium_fidelity_avg_vs_alpha_and_omega_simplified_square_wave_n=" + str(n) + "_(EC,EL,EJ)=" + str(
        [EC, EL, EJ]) + "_omega="
                + str(omega_list) + "_alpha=" + str(alpha_list) + "_start_state=" + str(start_state) + "_cycle_num="
                + str(cycle_num) + ".pdf")

    h5f = h5py.File("Plots/Fluxonium_fidelity_avg_vs_alpha_and_omega_simplified_square_wave_n=" + str(n) + "_(EC,EL,EJ)=" + str(
        [EC, EL, EJ]) + "_omega="
                + str(omega_list) + "_alpha=" + str(alpha_list) + "_start_state=" + str(start_state) + "_cycle_num="
                + str(cycle_num) + ".h5", 'w')
    h5f.create_dataset('data_1', data=end_fid_arr)


# Running functions

cycle_num = 50  # this is the number of drive cycles we evolve the system for
start_state = 5  # I initialize the system in an eigenstate of H_static = 4E_C n^2 +EJ/2 phi^2. This variable controls which eigenstate (where 0 is the ground state)
EL = 12.58
EC = 0.33
EJ = 12.58
n = 50  # This is the onsite hilbert space basis
k = n   # k defines the order to which cos(phi) is truncated, taking k=n truncates makes it exact for a given n

# Below we define alpha = A/omega to be the ratio between the drive amplitude A and drive frequency A
# You will need to make another directory called Plots for the plots to be saved to
# E_fid is calculated as <H_static(T)>/<H_static(0)> where T is at the end of a cycle
# to make things less noisy I often average E_fid over all the cycles we evolve over (noted with avg in the function name)
# fidelity is calculated as |<psi(T)|psi(0)|^2 (as per normal) and also averaged over the cycles considered.


#### This makes the graphs for E_fid vs alpha at constant omega
"""
omega = 17
alpha_list = [0, 6, 61]
# E_fid_vs_alpha_graph(n, k, EC, EL, EJ, omega, start_state, cycle_num, alpha_list)
E_avg_fid_vs_alpha_graph(n, k, EC, EL, EJ, omega, start_state, cycle_num, alpha_list)
"""

#### This makes the graphs for E_fid vs omega at constant alpha
"""
alpha = 2.7
omega_list = [1, 60, 62]
E_fid_vs_omega_graph(n, k, EC, EL, EJ, omega_list, start_state, cycle_num, alpha)
"""

#### This makes a color map for E_fid vs omega and alpha
"""
alpha_list = [0, 7, 71]
omega_list = [1, 60, 59]
E_avg_fid_vs_alpha_and_omega_heat_map(n, k, EC, EL, EJ, omega_list, start_state, cycle_num, alpha_list)
"""

#### This makes graphs for fidelity avg vs alpha
"""
omega = 34
alpha_list = [0, 6, 61]
# alpha = 2.5
# A = alpha*omega
# overlap_list, t_indices = evolve_fidelity_with_drive(n, k, cycle_num, EC, EL, EJ, A, omega, start_state)
# print(np.real(overlap_list))
fidelity_avg_vs_alpha_graph(n, k, EC, EL, EJ, omega, start_state, cycle_num, alpha_list)
"""

#### This makes a color map for fidelity avg vs omega and alpha
"""
alpha_list = [0, 6, 61]
omega_list = [1, 50, 49]
fid_avg_vs_alpha_and_omega_heat_map(n, k, EC, EL, EJ, omega_list, start_state, cycle_num, alpha_list)
"""


