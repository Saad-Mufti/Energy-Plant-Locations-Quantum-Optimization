import time
from math import log
import geopy.distance
import numpy as np
import pandas as pd
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from entropica_qaoa.qaoa.cost_function import (QAOACostFunctionOnQVM,
                                               QAOACostFunctionOnWFSim)
from entropica_qaoa.qaoa.parameters import ExtendedParams
from entropica_qaoa.utilities import (cluster_accuracy, distances_dataset,
                                      graph_from_hamiltonian,
                                      hamiltonian_from_distances,
                                      max_probability_bitstring, plot_graph)
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm
from scipy.optimize import minimize
from pyquil.api import QuantumComputer
import pyquil.api as api
from grove.pyqaoa.qaoa import QAOA
import pyquil as pq
import itertools

dataset = pd.read_csv('C:/Users/Saad Mufti/Documents/GitHub/Energy-Plant-Locations-Quantum-Optimization/EnergyAccessSurveyFull.csv', low_memory=False, encoding='ANSI')
dataset = dataset.set_index('Sr_ no_')
dataset = dataset[['GPS', 'ES3']]
dataset[['Elec_access']] = dataset[['ES3']] == 7 
without_elec = dataset.loc[dataset['Elec_access'] == True] 
without_elec = without_elec.drop_duplicates(subset='GPS')
# print(without_elec)
bools = list(without_elec['Elec_access'])

without_elec[['Latitude', 'Longitude']] = without_elec['GPS'].str.split(',', expand=True)
without_elec = without_elec[['Latitude', 'Longitude']]
dataset = dataset.drop_duplicates(subset='GPS') 
dataset.append(without_elec)

# print('Dataset has', len(dataset))

without_elec = without_elec.sample(3) #Can't actually deal with many points, so only limiting to 3 (random) ones
distance_matrix = np.array(without_elec[['Latitude', 'Longitude']])

dist = pd.DataFrame(distances_dataset(without_elec.values, 
                                    lambda u, v: geopy.distance.distance(u, v).kilometers), index=without_elec.index, columns=without_elec.index)

distance_matrix = np.array(dist)
print(distance_matrix)


qvm = api.get_qc('9q-qvm') # QC Emulator
qvm.compiler.client.timeout = 240 # Adjust this for different sized distance matrix

starting_node = 0
reduced_number_of_nodes = len(distance_matrix)
number_of_qubits = reduced_number_of_nodes ** 2
qubits = list(range(number_of_qubits)) # Just an index of the qubits being used (ie qubit #0, #1, etc)

def create_phase_separator():
        """
        Creates phase-separation operators (aka cost hamiltonian), which depend on the objective function.
        """
        cost_operators = []
        reduced_distance_matrix = np.delete(distance_matrix, starting_node, axis=0)
        reduced_distance_matrix = np.delete(reduced_distance_matrix, starting_node, axis=1)
        reduced_number_of_nodes = len(reduced_distance_matrix)
        number_of_qubits = reduced_number_of_nodes ** 2

        for t in range(reduced_number_of_nodes - 1):
            for city_1 in range(reduced_number_of_nodes):
                for city_2 in range(reduced_number_of_nodes):
                    if city_1 != city_2:
                        distance = reduced_distance_matrix[city_1, city_2] 
                        qubit_1 = t * (reduced_number_of_nodes) + city_1
                        qubit_2 = (t + 1) * (reduced_number_of_nodes) + city_2
                        cost_operators.append(PauliTerm("Z", qubit_1, distance) * PauliTerm("Z", qubit_2))

        costs_to_starting_node = np.delete(distance_matrix[:, starting_node], starting_node)
        for city in range(reduced_number_of_nodes):
            distance_from_0 = -costs_to_starting_node[city]
            qubit = city
            cost_operators.append(PauliTerm("Z", qubit, distance_from_0))

        for city in range(reduced_number_of_nodes):
            distance_from_0 = -costs_to_starting_node[city]
            qubit = number_of_qubits - (reduced_number_of_nodes) + city
            cost_operators.append(PauliTerm("Z", qubit, distance_from_0))

        phase_separator = [PauliSum(cost_operators)]
        return phase_separator

def create_mixing_hamiltonian():
        mixer_operators = []

        for t in range(reduced_number_of_nodes - 1):
            for city_1 in range(reduced_number_of_nodes):
                for city_2 in range(reduced_number_of_nodes):
                    c1 = city_1
                    c2 = city_2

                    first_part = 1
                    first_part *= s_plus(c1, t)
                    first_part *= s_plus(c2, t+1)
                    first_part *= s_minus(c1, t+1)
                    first_part *= s_minus(c2, t)

                    second_part = 1
                    second_part *= s_minus(c1, t)
                    second_part *= s_minus(c2, t+1)
                    second_part *= s_plus(c1, t+1)
                    second_part *= s_plus(c2, t)

                    mixer_operators.append(first_part + second_part)

        return mixer_operators

def s_plus(city, time):
        qubit = time * (reduced_number_of_nodes) + city
        return PauliTerm("X", qubit) + PauliTerm("Y", qubit, 1j)


def s_minus(city, time):
        qubit = time * (reduced_number_of_nodes) + city
        return PauliTerm("X", qubit) - PauliTerm("Y", qubit, 1j)

def binary_to_decimal(binary_list):
    """Turns the bitstring into an list that represents the order of points"""
    num = int(np.sqrt(len(binary_list)))
    decimal = []
    for i in range(num):
        for j in range(num):
            if binary_list[num * i + j] == 1:
                decimal.append(j)

    return decimal

def create_initial_state_program():
        initial_state_program = pq.Program()
        vector_of_states = np.zeros(2**number_of_qubits)
        list_of_possible_states = []
        initial_order = range(0, reduced_number_of_nodes)
        all_permutations = [list(x) for x in itertools.permutations(initial_order)]
        for permutation in all_permutations:
            coding_of_permutation = 0
            for i in range(len(permutation)):
                coding_of_permutation += 2**(i * (reduced_number_of_nodes) + permutation[i])
            vector_of_states[coding_of_permutation] = 1
        initial_state_program = create_arbitrary_state(vector_of_states)
        return initial_state_program

def get_solution_for_full_array(reduced_solution):
        full_solution = reduced_solution
        for i in range(len(full_solution)):
            if full_solution[i] >= starting_node:
                full_solution[i] += 1
        full_solution.insert(0, starting_node)
        return full_solution

def calculate_solution():
        betas, gammas = qaoa_inst.get_angles() # Performs VQE algorithm on betas + gammas
        print("betas", betas)
        print("gammas", gammas)
        most_frequent_string, sampling_results = qaoa_inst.get_string(betas, gammas, samples=5000) 
        reduced_solution = binary_to_decimal(most_frequent_string)
        full_solution = get_solution_for_full_array(reduced_solution)
        solution = full_solution
        all_solutions = sampling_results.keys()
        distribution = {}
        for sol in all_solutions:
            reduced_sol = binary_to_decimal(sol)
            full_sol = get_solution_for_full_array(reduced_sol)
            distribution[tuple(full_sol)] = sampling_results[sol]
        distribution = distribution
        return solution


minimizer_kwargs = {'method': 'Nelder-Mead', 'options': {'ftol': 1.0e-3, 'xtol': 1.0e-3,'disp': False}}
def print_fun(x):
    print(x)
vqe_option = {'disp': print_fun, 'return_all': True, 'samples': None}

qaoa_inst = QAOA(qvm, qubits, 
                    steps=2, 
                    cost_ham=create_phase_separator(),
                    ref_ham= create_mixing_hamiltonian(), 
                    driver_ref=create_initial_state_program(),
                    minimizer=minimize,
                    minimizer_kwargs=minimizer_kwargs,
                    rand_seed=None,
                    vqe_options=vqe_option, 
                    store_basis=True)
                    
print("Solution", calculate_solution())