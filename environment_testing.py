import numpy as np

from tqdm import tqdm
from ase.neighborlist import neighbor_list
from timeit import default_timer as timer

def tester(env, images):

	results = []

	print('Running all tests.')

	for atoms in tqdm(images):

		### Get base information ###

		cell = atoms.get_cell()
		positions = atoms.get_positions()
		n_atoms = len(atoms)

		### Get reference data ###

		idx_i, idx_j, idx_S = neighbor_list('ijS', atoms, env.cutoff, self_interaction=False)

		uidx, n_nbh = np.unique(idx_i, return_counts=True)
		max_n_nbh = np.max(n_nbh)

		nbh_range = np.tile(np.arange(max_n_nbh,dtype=np.int)[np.newaxis], (n_atoms,1))

		mask = np.zeros((n_atoms,max_n_nbh), dtype=np.bool)
		mask[uidx,:] = nbh_range < np.tile(n_nbh[:,np.newaxis], (1,max_n_nbh))

		nbh = -np.ones((n_atoms,max_n_nbh), dtype=np.int)
		nbh[mask] = idx_j

		offsets = np.zeros((n_atoms,max_n_nbh,3), dtype=np.float)
		offsets[mask] = idx_S

		distances = np.linalg.norm(positions[nbh[:,:],:] - positions[:,None,:] + offsets @ cell, 2, 2)

		sorted_nbh = np.sort(nbh, axis=-1)
		sorted_distances = np.sort(distances, axis=-1)

		### Get environment provider data ###

		env_nbh, env_offsets = env.get_environment(atoms)
		env_nbh = env_nbh.astype(np.int)
		env_offsets = env_offsets.astype(np.float)

		env_n_nbh = np.sum(env_nbh >= 0, 1)

		env_distances = np.linalg.norm(positions[env_nbh[:,:],:] - positions[:,None,:] + env_offsets @ cell, 2, 2)

		env_sorted_nbh = np.sort(env_nbh, axis=-1)
		env_sorted_distances = np.sort(env_distances, axis=-1)

		### Compare the environment provider against the reference ###

		# Test 1: number of neighbors for each atom.
		t1 = np.allclose(n_nbh, env_n_nbh)

		# Test 2: shape of neighbors array.
		t2 = nbh.shape == env_nbh.shape

		# Test 3: sorted neighbor index for each atom.
		t3 = np.allclose(sorted_nbh, env_sorted_nbh)

		# Test 4: sorted interatomic distances for all neighbors.
		t4 = np.allclose(sorted_distances, env_sorted_distances)

		results.append([t1, t2, t3, t4])

	results = np.array(results)

	if results.all():
		print('All tests passed for all images!')
		return np.array(results)
	else:
		print('Tests failed for:')
		for i in np.argwhere(results==False):
			print('   image %i, test %i' % tuple(i + [0,1]))

		return np.array(results)

def benchmarker(env, images, loops=10):

	time = []
	print('Running benchmarks!')

	for atoms in tqdm(images):
		
		start = timer()
		for _ in range(loops):
			env.get_environment(atoms)
		end = timer()

		time.append((end - start) / loops)

	return np.array(time)
