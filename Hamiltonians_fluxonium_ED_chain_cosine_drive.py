import numpy as np
import math
from quspin.operators import hamiltonian
from scipy.special import jv, struve



##### Subfunctions used to create the Hamiltonians. The outputs of all these functions are unlikely to be used elsewhere

def cos_sin_drive(t, omega, A):  # Defines the cosine of the sin drive
    return np.cos((A/omega)*np.sin(omega*t))


def sin_sin_drive(t, omega, A):  # Defines the sine of the sin drive
    return np.sin((A/omega)*np.sin(omega*t))


def make_ordered_exp_b(k, site_i, pm_str, phi_coeff, basis):
    """Calculates e^(i*phi_coeff*{b^+, b}). The variable pm_str = "+", "-" defines whether it is b^dagger or b"""
    static_exp_phi = []
    for i in range(k + 1):
        j = i
        pm_j = pm_str * j
        static_qubit_exp_list = []
        interactj = [site_i] * j
        interactj = [(1/math.factorial(j))*((1j*phi_coeff)**j)] + interactj
        static_qubit_exp_list.append(interactj)
        static_exp_phi.append([pm_j, static_qubit_exp_list])
    H_exp_b = hamiltonian(static_exp_phi, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_exp_b


def make_ordered_exp_b_cos_t(k, site_i, pm_str, phi_coeff, omega, A, basis):
    dynamic_exp_phi = []
    for i in range(k + 1):
        j = i
        pm_j = pm_str * j
        dynamic_qubit_exp_list = []
        interactj = [site_i] * j
        interactj = [(1/math.factorial(j))*((1j*phi_coeff)**j)] + interactj
        dynamic_qubit_exp_list.append(interactj)
        dynamic_exp_phi.append([pm_j, dynamic_qubit_exp_list, cos_sin_drive, [omega, A]])
    H_exp_b = hamiltonian([], dynamic_exp_phi, dtype=np.complex64, basis=basis, check_herm=False)
    return H_exp_b


def make_ordered_exp_b_sin_t(k, site_i, pm_str, phi_coeff, omega, A, basis):
    dynamic_exp_phi = []
    for i in range(k + 1):
        j = i
        pm_j = pm_str * j
        dynamic_qubit_exp_list = []
        interactj = [site_i] * j
        interactj = [(1/math.factorial(j))*((1j*phi_coeff)**j)] + interactj
        dynamic_qubit_exp_list.append(interactj)
        dynamic_exp_phi.append([pm_j, dynamic_qubit_exp_list, sin_sin_drive, [omega, A]])
    H_exp_b = hamiltonian([], dynamic_exp_phi, dtype=np.complex64, basis=basis, check_herm=False)
    return H_exp_b


def cos_cos_t_ordered_Hamiltonian(k, L, phi_coeff, omega, A, basis):
    H_list = []
    for site_i in range(L):
        H_exp_bp_cos_t = make_ordered_exp_b_cos_t(k, site_i, '+', phi_coeff, omega, A, basis)
        H_exp_bm = make_ordered_exp_b(k, site_i, '-', phi_coeff, basis)
        H_exp_mbp_cos_t = make_ordered_exp_b_cos_t(k, site_i, '+', -1*phi_coeff, omega, A, basis)
        H_exp_mbm = make_ordered_exp_b(k, site_i, '-', -1*phi_coeff, basis)
        non_comuting_coeff = np.exp(-1*(phi_coeff**2)/2)/2
        ### Note that the next line calculates cos by writing it in terms of exponentials then using BCH to normal order them
        ### The normal ordering is done before we truncate the Hilbert space so the order of operations is correct
        ### In each term, one of the exponetials is a dynamic Hamiltonian which is multiplied by appropriate cosine drive
        cos_cos_t_ordered_H_i = non_comuting_coeff*(H_exp_bp_cos_t*H_exp_bm + H_exp_mbp_cos_t*H_exp_mbm)
        H_list.append(cos_cos_t_ordered_H_i)
    cos_cos_t_ordered_H = sum(H_list)
    return cos_cos_t_ordered_H


def sin_sin_t_ordered_Hamiltonian(k, L, phi_coeff, omega, A, basis):
    H_list = []
    for site_i in range(L):
        H_exp_bp_sin_t = make_ordered_exp_b_sin_t(k, site_i, '+', phi_coeff, omega, A, basis)
        H_exp_bm = make_ordered_exp_b(k, site_i, '-', phi_coeff, basis)
        H_exp_mbp_sin_t = make_ordered_exp_b_sin_t(k, site_i, '+', -1 * phi_coeff, omega, A, basis)
        H_exp_mbm = make_ordered_exp_b(k, site_i, '-', -1 * phi_coeff, basis)
        non_comuting_coeff = np.exp(-1 * (phi_coeff ** 2) / 2) / (2 * 1j)
        sin_sin_t_ordered_H_i = non_comuting_coeff * (H_exp_bp_sin_t * H_exp_bm - H_exp_mbp_sin_t * H_exp_mbm)
        H_list.append(sin_sin_t_ordered_H_i)
    sin_sin_t_ordered_H = sum(H_list)
    return sin_sin_t_ordered_H



##### Functions that calculate useful Hamiltonians

def n_i(site_i, n_coeff, basis):
    """Calculates n_i"""
    static_H_n_i = [["+", [[n_coeff, site_i]]], ["-", [[-n_coeff, site_i]]]]
    H_n_i = hamiltonian(static_H_n_i, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_n_i


def phi_i(site_i, phi_coeff, basis):
    """Calculates phi_i"""
    static_H_phi_i = [["+", [[phi_coeff, site_i]]], ["-", [[phi_coeff, site_i]]]]
    H_phi_i = hamiltonian(static_H_phi_i, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_phi_i


def cos_ordered(k, L, phi_coeff, basis):
    """Calculates sum cos(phi) in the normal ordered expansion"""
    H_list = []
    for site_i in range(L):
        H_exp_bp = make_ordered_exp_b(k, site_i,'+', phi_coeff, basis)
        H_exp_bm = make_ordered_exp_b(k, site_i, '-', phi_coeff, basis)
        H_exp_mbp = make_ordered_exp_b(k, site_i, '+', -1*phi_coeff, basis)
        H_exp_mbm = make_ordered_exp_b(k, site_i, '-', -1*phi_coeff, basis)
        non_comuting_coeff = np.exp(-1*(phi_coeff**2)/2)/2
        cos_ordered_H_i = non_comuting_coeff*(H_exp_bp*H_exp_bm + H_exp_mbp*H_exp_mbm)
        H_list.append(cos_ordered_H_i)
    cos_ordered_H = sum(H_list)
    return cos_ordered_H


def driven_cos_ordered(k, L, phi_coeff, omega, A, basis):
    """Calculates sum cos(phi - A*sin(omega*t)) in the normal ordered basis"""
    return cos_cos_t_ordered_Hamiltonian(k, L, phi_coeff, omega, A, basis) + sin_sin_t_ordered_Hamiltonian(k, L, phi_coeff, omega, A, basis)


def n2(L, n_coeff, basis):
    """Calculates sum n^2"""
    coeff_and_loc_list_p = []
    coeff_and_loc_list_m = []
    for m in range(L):
        interactj = [m] * 2
        interactj_p = [n_coeff ** 2] + interactj
        interactj_m = [-n_coeff ** 2] + interactj
        coeff_and_loc_list_p.append(interactj_p)
        coeff_and_loc_list_m.append(interactj_m)
    static_n2 = [["++", coeff_and_loc_list_p], ["--", coeff_and_loc_list_p], ["+-", coeff_and_loc_list_m], ["-+", coeff_and_loc_list_m]]
    H_n2 = hamiltonian(static_n2, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_n2


def phi2(L, phi_coeff, basis):
    """Calculates sum phi^2"""
    coeff_and_loc_list_p = []
    for m in range(L):
        interactj = [m] * 2
        interactj_p = [phi_coeff ** 2] + interactj
        coeff_and_loc_list_p.append(interactj_p)
    static_phi2 = [["++", coeff_and_loc_list_p], ["--", coeff_and_loc_list_p], ["+-", coeff_and_loc_list_p], ["-+", coeff_and_loc_list_p]]
    H_phi2 = hamiltonian(static_phi2, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_phi2


def n(L, n_coeff, basis):
    """Calculates sum n"""
    coeff_and_loc_list_p = []
    coeff_and_loc_list_m = []
    for m in range(L):
        interactj = [m] * 1
        interactj_p = [n_coeff] + interactj
        interactj_m = [-n_coeff] + interactj
        coeff_and_loc_list_p.append(interactj_p)
        coeff_and_loc_list_m.append(interactj_m)
    static_n = [["+", coeff_and_loc_list_p], ["-", coeff_and_loc_list_m]]
    H_n = hamiltonian(static_n, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_n


def phi(L, phi_coeff, basis):
    """Calculates sum phi"""
    coeff_and_loc_list_p = []
    for m in range(L):
        interactj = [m] * 1
        interactj_p = [phi_coeff] + interactj
        coeff_and_loc_list_p.append(interactj_p)
    static_phi = [["+", coeff_and_loc_list_p], ["-", coeff_and_loc_list_p]]
    H_phi = hamiltonian(static_phi, [], dtype=np.complex64, basis=basis, check_herm=False)
    return H_phi


def nn(L, n_coeff, basis):
    """Calculates sum n_i*n_{i+1}. Returns 0 if L = 1"""
    if L == 0:
        return 0*n(L, n_coeff, basis)
    else:
        coeff_and_loc_list_p = []
        coeff_and_loc_list_m = []
        for m in range(L-1):
            interactj = [m, m+1]
            interactj_p = [n_coeff**2] + interactj
            interactj_m = [-n_coeff**2] + interactj
            coeff_and_loc_list_p.append(interactj_p)
            coeff_and_loc_list_m.append(interactj_m)
        static_nn = [["++", coeff_and_loc_list_p], ["--", coeff_and_loc_list_p], ["+-", coeff_and_loc_list_m], ["-+", coeff_and_loc_list_m]]
        H_nn = hamiltonian(static_nn, [], dtype=np.complex64, basis=basis, check_herm=False)
        return H_nn


def magnus_residual(k, L, EC, EJ, n_coeff, phi_coeff, omega, A, basis):
    """Returns the residual Hamiltonian after subtracting the bosonic part to first order Magnus. Calculated in the normal ordered expansion"""
    H_n_cos_list = []
    H_cos_n_list = []
    for site_i in range(L):
        H_exp_bp = make_ordered_exp_b(k, site_i, '+', phi_coeff, basis)
        H_exp_bm = make_ordered_exp_b(k, site_i, '-', phi_coeff, basis)
        H_exp_mbp = make_ordered_exp_b(k, site_i, '+', -1 * phi_coeff, basis)
        H_exp_mbm = make_ordered_exp_b(k, site_i, '-', -1 * phi_coeff, basis)
        non_comuting_coeff = np.exp(-1 * (phi_coeff ** 2) / 2) / 2
        cos_ordered_H_i = non_comuting_coeff * (H_exp_bp * H_exp_bm + H_exp_mbp * H_exp_mbm)
        H_n_i = n_i(site_i, n_coeff, basis)
        H_n_cos_list.append(H_n_i * cos_ordered_H_i)
        H_cos_n_list.append(cos_ordered_H_i * H_n_i)
    n_cos_ordered_H = sum(H_n_cos_list)
    cos_n_ordered_H = sum(H_cos_n_list)
    H_cos_phi = cos_ordered(k, L, phi_coeff, basis)
    alpha = A/omega
    u = 2*np.pi*EC*EJ*struve(0, alpha)
    return -jv(0, alpha)*EJ*H_cos_phi + (u/omega)*(n_cos_ordered_H + cos_n_ordered_H)


"""
##### Examples of constants and what they refer to

EC = 0.33
EL = 1
EJ = 12.58
V = 0.05
omega = 15
A = 2.4048*omega

L = 2  # system length
n = 10  # local Hilbert space dimension
k = n  # the power to which we expand the cosine terms (k>n is equivalent to k=n)  

n_coeff = (1j / 2) * (EL / (2 * EC)) ** (1 / 4)  # the coefficient of n: n = n_coeff(b^+ - b)
phi_coeff = ((2 * EC) / EL) ** (1 / 4)  # the coefficient of phi: phi = phi_coeff(b^+ + b)
basis = boson_basis_1d(L, sps=n)  # the bosonic basis used throughout


##### Examples of commonly used sub Hamiltonians and the equations they represent

# n_i (site_i determines which n_i in the chain)
H_n_i = H.n_i(site_i, n_coeff, basis)

# phi_i (site_i determines which phi_i in the chain
H_phi_i = H.phi_i(site_i, phi_coeff, basis)

# sum_i n_i
H_n = H.n(L, n_coeff, basis)

# sum_i phi_i
H_phi = H.phi(L, phi_coeff, basis)

# sum_i 4*EC*n_i^2
H_EC = (4 * EC) * H.n2(L, n_coeff, basis)

# sum_i (EL/2)*phi_i^2
H_EL = (EL / 2) * H.phi2(L, phi_coeff, basis)

# sum_i -EJ*cos(phi_i - A sin(omega t)). Note that this Hamiltonian is time dependent
H_EJ_driven = -EJ*H.driven_cos_ordered(k, L, phi_coeff, omega, A, basis)

# sum_i V*n_i*n_{i+1}. Note that if L=0, this will simply return 0. We do not use periodic boundary conditions
H_V = V*H.nn(L, n_coeff, basis)

# sum_i first order magnus terms
H_first_order_magnus = H.magnus_residual(k, L, EC, EJ, n_coeff, phi_coeff, omega, A, basis)


##### Examples of commonly used total Hamiltonians

# total driven fluxonium chain (time dependent)
H_total_driven = H_EC + H_EL + H_EJ_driven + H_V

# total time independent fluxonium chain (up to first order magnus)
H_total_magnus = H_EC + H_EL + H_first_order_magnus + H_V
"""
