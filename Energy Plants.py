import pandas as pd
# import networkx as nx
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm
from entropica_qaoa.utilities import hamiltonian_from_distances
from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnQVM, QAOACostFunctionOnWFSim
from entropica_qaoa.qaoa.parameters import ExtendedParams
from scipy.optimize import minimize
import time
import numpy as np
from math import log
from entropica_qaoa.utilities import cluster_accuracy, max_probability_bitstring
from entropica_qaoa.utilities import distances_dataset
from scipy.spatial.distance import pdist
import geopy.distance

dataset = pd.read_csv('C:/Users/Saad Mufti/Documents/GitHub/Energy-Plant-Locations-Quantum-Optimization/EnergyAccessDataset.csv', encoding='ANSI')
dataset = dataset.set_index('Sr_ no_')
dataset = dataset.drop_duplicates(subset='GPS')
#TODO: Use bigger dataset instead so can get trues and falses. Then apply algo
# bools = dataset
# bools[['Elec_access']] = dataset[['ES3']] != 7 
# bools = list(bools['Elec_access'])
# print(bools)
dataset = dataset[['GPS']]

dataset[['Latitude', 'Longitude']] = dataset['GPS'].str.split(',', expand=True)
dataset = dataset.drop(['GPS'], axis=1)
# print(dataset)

distance_matrix = np.array(dataset[['Latitude', 'Longitude']])
# print(distance_matrix)

# m_dist = pdist(distance_matrix, lambda u, v: geopy.distance.distance(u, v).kilometers)

# print(m_dist)

dist = pd.DataFrame(distances_dataset(dataset.values, 
                                    lambda u, v: geopy.distance.distance(u, v).kilometers), index=dataset.index, columns=dataset.index)
print(dist)
# print(dist.iloc[0,:])

hamiltonian = hamiltonian_from_distances(dist)
timesteps = 3
iterations = 500
n_qubits = 7 #10
betas = [round(val,1) for val in np.random.rand(timesteps*n_qubits)]
gammas_singles = [round(val,1) for val in np.random.rand(0)] 
gammas_pairs = [round(val,1) for val in np.random.rand(timesteps*len(hamiltonian))]


hyperparameters = (hamiltonian, timesteps)
parameters = (betas, gammas_singles, gammas_pairs)

params = ExtendedParams(hyperparameters, parameters)

sim = WavefunctionSimulator()
cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                        params=params,
                                        sim=sim,
                                        enable_logging=True)

t0 = time.time()


print('Run complete!\n','Runtime:','{:.3f}'.format(time.time()-t0))

def run_qaoa(hamiltonian, params, timesteps, max_iters, init_state=None):
    cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                            params=params,
                                            initial_state=init_state)
    res = minimize(cost_function, params.raw(), tol=1e-3, method='Cobyla',
                          options={"maxiter" : max_iters})

    return cost_function.get_wavefunction(params.raw()), res

wave_func, res = run_qaoa(hamiltonian, params, timesteps=3, max_iters=1500)

wave_func = cost_function.get_wavefunction(params.raw())
lowest = max_probability_bitstring(wave_func.probabilities())

true_clusters = [1 if val else 0 for val in labels]
acc = cluster_accuracy(lowest,true_clusters)

print(res)