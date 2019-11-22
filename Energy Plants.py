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

dataset = pd.read_csv('C:/Users/Saad Mufti/Documents/GitHub/Energy-Plant-Locations-Quantum-Optimization/EnergyAccessSurveyFull.csv', low_memory=False, encoding='ANSI')
dataset = dataset.set_index('Sr_ no_')
dataset = dataset[['GPS', 'ES3']]
dataset[['Elec_access']] = dataset[['ES3']] == 7 
without_elec = dataset.loc[dataset['Elec_access'] == True] 
without_elec = without_elec.drop_duplicates(subset='GPS')
print(without_elec)

dataset = dataset.drop_duplicates(subset='GPS') 
dataset.append(without_elec)

print('Dataset has', len(dataset))

bools = list(without_elec['Elec_access'])
dataset = dataset[['GPS']]
dataset_sample = pd.concat([without_elec[['GPS']], dataset.sample(8)])
print(dataset_sample)
dataset_sample[['Latitude', 'Longitude']] = dataset_sample['GPS'].str.split(',', expand=True)
dataset_sample = dataset_sample.drop(['GPS'], axis=1)
print(dataset_sample)

distance_matrix = np.array(dataset_sample[['Latitude', 'Longitude']])

dist = pd.DataFrame(distances_dataset(dataset_sample.values, 
                                    lambda u, v: geopy.distance.distance(u, v).kilometers), index=dataset_sample.index, columns=dataset_sample.index)
print(dist)

hamiltonian = hamiltonian_from_distances(dist)
timesteps = 3
iterations = 500
n_qubits = 16 
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

def run_qaoa(hamiltonian, params, timesteps, max_iters, init_state=None):
    cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                            params=params,
                                            initial_state=init_state)
    res = minimize(cost_function, params.raw(), tol=1e-3, method='Cobyla',
                          options={"maxiter" : max_iters})

    return cost_function.get_wavefunction(params.raw()), res

t0 = time.time()
wave_func, res = run_qaoa(hamiltonian, params, timesteps=3, max_iters=2500)

print('Run complete!\n','Runtime:','{:.3f}'.format(time.time()-t0))
wave_func = cost_function.get_wavefunction(params.raw())
lowest = max_probability_bitstring(wave_func.probabilities())

true_clusters = [1 if val else 0 for val in bools]
print(cluster_accuracy(lowest,true_clusters))

print(res)