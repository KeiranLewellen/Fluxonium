# importing all the packagaes
from __future__ import print_function, division
import sys, os

# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, qspin_path)
os.environ['MKL_DEBUG_CPU_TYPE'] = '63'  # AVX instructions; good for any reasonably recent
from quspin.operators import hamiltonian, exp_op, commutator  # Hamiltonians and operators# operators
from quspin.basis import boson_basis_1d  # Hilbert space boson basis
# from quspin.tools.Floquet import Floquet, Floquet_t_vec  # Floquet Hamiltonian
from quspin.tools.Floquet import Floquet, Floquet_t_vec  # Floquet Hamiltonian
import numpy as np  # generic math functions
import matplotlib.pyplot as plt
from quspin.tools.measurements import obs_vs_time

#################################################################################################

# Parameters

# n_jobs_floquet = 63  # number of threads for parallel computing
# L = 8  # system size
# N_cycles = 1000

# Defines main class


class FluxoniumED:
    def __init__(self, L, n, EJ, EL, EC, Omega, A):
        self.Omega = Omega
        self.Eb_t = None
        self.N_cycles = None
        self.psi0 = None
        self.t1 = None  # Stroboscopic indices

        n_coeff = 1j/2 #1j*(1 / 2) * ((EJ / (2 * EC)) ** 0.25)
        print(n_coeff)
        phi_coeff = 1 #(((2 * EC) / EJ) ** 0.25)
        print(n_coeff*phi_coeff)

        # Setting up the Hamiltonian

        basis = boson_basis_1d(L, Nb=None, sps=n)  # particle non conserving basis, n states per site

        pot_n1 = [[A*n_coeff, i] for i in range(L)]  # for the floquet drive
        pot_n2 = [[-A*n_coeff, i] for i in range(L)]  # for the floquet drive
        pot_phi = [[-EL*A*phi_coeff, i] for i in range(L)]  # for the floquet drive
        print("pot_n1: ", pot_n1)
        print("pot_phi: ", pot_phi)

        interact_EL = [[(EL/2)*(phi_coeff**2), i, i] for i in range(L)]  # interaction term for EL

        # operators defined here

        dynamic_n = [["+", pot_n1, self.square_drive, []], ["-", pot_n2, self.square_drive, []]]
        dynamic_phi = [["+", pot_phi, self.triangular_drive, []], ["-", pot_phi, self.triangular_drive, []]]

        static_EL = [['+-', interact_EL], ['-+', interact_EL], ['++', interact_EL], ['--', interact_EL]]


        # Defining the static hamiltonian

        H_EL = hamiltonian(static_EL, [], dtype=np.float64, basis=basis)

        self.H_bosonic = H_EL
        self.H_fluxonium = H_EL

        # Floquet Hamiltonian for dynamic evolution quasinergy/ floquet modes solution

        H_n_drive = hamiltonian([], dynamic_n, dtype=np.complex64, basis=basis)
        H_phi_drive = hamiltonian([], dynamic_phi, dtype=np.complex64, basis=basis)
        self.H_fluxonium_driven = self.H_fluxonium + H_n_drive + H_phi_drive

    def square_drive(self, t):  # Defines a square drive used to drive n
        return np.sign(np.sin(self.Omega * t))

    def triangular_drive(self, t):  # Defines the integral of the square drive -- used to drive phi
        return (2 * np.pi / self.Omega) * np.abs(t * self.Omega / (2 * np.pi) - np.floor(t * self.Omega / (2 * np.pi) + 1/2))

    def bosonic_excited_state(self, k):
        E2, V2 = self.H_bosonic.eigh()  # eigenvalue and eigenstate of H_bosonic
        psi0 = V2[:, k]  # initialization, k means kth excited state
        print("Energies: ", E2)
        return psi0

    def floquet_time_evolve(self, N_cycles, psi0, len_T=1):
        t1 = Floquet_t_vec(self.Omega, N_cycles, len_T=len_T)  # change N cycle, len_T=1 means 1 cycle per evolution, quspin default implement 100 trotter points per cycle
        # print("t1: ", t1)
        # print(len(t1))
        psi_t_gen = self.H_fluxonium_driven.evolve(psi0, t1.i, t1.vals, iterate=True)  # look at this line -- this does not do the computations just sets them up
        return psi_t_gen, t1

    def fidelity_vs_time(self, N_cycles, psi0, len_T=1):
        if not (np.all(self.N_cycles == N_cycles) and np.all(self.psi0 == psi0) and len(self.t1) == (len_T*N_cycles+1)):
            psi_t_gen, t1 = self.floquet_time_evolve(N_cycles, psi0, len_T=len_T)
            Eb_t = obs_vs_time(psi_t_gen, t1.vals, dict(E=self.H_bosonic), return_state=True)  # calculating the Eb observable -- this actually does the computation, sometimes faster to calculate psi once using something like this and then calculate many observables from it
            self.Eb_t = Eb_t
            self.N_cycles = N_cycles
            self.psi0 = psi0
            self.t1 = t1
        psi_t = self.Eb_t["psi_t"]  # finds the state after being computed

        overlap_list = []
        for j in range(N_cycles + 1):
            overlap = np.vdot(psi0, psi_t[:, j])
            overlap_abs = abs(overlap)
            overlap_list.append(overlap_abs)
            # print(overlap_abs)
        return overlap_list, self.t1

    def E_bosonic_vs_time(self, N_cycles, psi0, len_T=1):
        Eb_i = self.H_bosonic.expt_value(psi0)  # initial E_bosonic
        print("Eb_i ", Eb_i)
        if not (np.all(self.N_cycles == N_cycles) and np.all(self.psi0 == psi0) and len(self.t1) == (len_T*N_cycles+1)):
            psi_t_gen, t1 = self.floquet_time_evolve(N_cycles, psi0, len_T=len_T)
            Eb_t = obs_vs_time(psi_t_gen, t1.vals, dict(E=self.H_bosonic), return_state=True)  # calculating the E_bosonic observable -- this actually does the computation, sometimes faster to calculate psi once using something like this and then calculate many observables from it
            # print(len(Eb_t["E"]))
            self.Eb_t = Eb_t
            self.N_cycles = N_cycles
            self.psi0 = psi0
            self.t1 = t1
        Eb_norm_t = self.Eb_t["E"] / Eb_i
        # print(Eb_norm_t)
        # Eb_norm_time_avg = np.sum(Eb_norm_t) / N_cycles
        # print("Time averaged Eb_norm: ", Eb_norm_time_avg)
        if np.all(np.imag(Eb_norm_t)) < 10**(-5):
            print("Eb_norm_t: ", np.real(Eb_norm_t))
            return np.real(Eb_norm_t), self.t1
        else:
            print("Imaginary value rather large: ", np.imag(Eb_norm_t))
            return None


# Parameters

L = 1
n = 101  # Hilbert space at each site
N_cycles = 250  # number of stroboscopic cycles we want to evolve
len_T = 1  # number of points per stroboscopic cycle

EJ = 12.58
#EJ = 5
#EL = 20
EL = 1
#EL = 12.58
EC = 0.33


# Omega = 15
Omega = 2*np.pi
A = 2*Omega
#A = 2.5*Omega
#A = 0
#A = 1.5*Omega

k = 5

Flux1 = FluxoniumED(L, n, EJ, EL, EC, Omega, A)
#psi0 = Flux1.bosonic_FES()
#psi0 = Flux1.bosonic_GS()
psi0 = Flux1.bosonic_excited_state(k)
E_list, time_steps1 = Flux1.E_bosonic_vs_time(N_cycles, psi0, len_T=len_T)
overlap_list, time_steps = Flux1.fidelity_vs_time(N_cycles, psi0, len_T=len_T)
print(overlap_list)
#print(time_steps)

print([Flux1.triangular_drive(t) for t in np.linspace(0, 3, 31)])
print([Flux1.square_drive(t) for t in np.linspace(0, 3, 31)])

#print(Flux1.triangular_drive(np.pi/Omega))
#print(1*np.pi/Omega)
#print(Flux1.triangular_drive(2*np.pi/Omega))

print("Calculated!")

# plotting the Nd(t) observabele


plt.plot(E_list, marker='.', markersize=0.05, label="$S=1/2$")
plt.title("Fluxonium no cos with drive")
plt.ylabel("Eb(t)/Eb(0)", fontsize=20)
plt.xlabel("$t/T$", fontsize=20)
plt.grid()
plt.savefig("Plots/Fluxonium_with_drive,no_cos,E_bosonic_vs_time,psi0="+str(k)+",len_T="+str(len_T)+",L="+str(L)+",dim="+str(n)+",EJ="+str(EJ)+",EC="+str(EC)+",EL="
            + str(EL)+",Omega="+str(Omega)+",A="+str(A)+",N_cycles="+str(N_cycles)+".pdf")
plt.savefig("Plots/Fluxonium_with_drive,no_cos,E_bosonic_vs_time,psi0="+str(k)+",len_T="+str(len_T)+",L="+str(L)+",dim="+str(n)+",EJ="+str(EJ)+",EC="+str(EC)+",EL="
            + str(EL)+",Omega="+str(Omega)+",A="+str(A)+",N_cycles="+str(N_cycles)+".png")
