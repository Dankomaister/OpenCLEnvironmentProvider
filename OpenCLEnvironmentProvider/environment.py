import os
import numpy as np
import pyopencl as cl

from schnetpack.environment import BaseEnvironmentProvider

class OpenCLEnvironmentProvider(BaseEnvironmentProvider):
	"""docstring for OpenCLEnvironmentProvider"""

	def __init__(self, cutoff, platform=0, number_density=0.25, cache_dir=None):
		"""
		Args:
			cutoff (float): the cutoff inside which atoms are considered pairs
		"""
		self.cutoff = np.float32(cutoff)
		self.max_nbh = np.uint32(np.ceil(4/3*np.pi*cutoff**3 * number_density))

		self.kernel_code = ''.join(open(os.path.dirname(os.path.realpath(__file__)) + '/neighbor_list_kernel.cl', 'r').readlines())
		self.kernel = None
		self.cache_dir = cache_dir
		self.platform = platform
		self.mf = cl.mem_flags

	def initialize(self):
		devices = cl.get_platforms()[self.platform].get_devices()
		self.ctx = cl.Context(devices)
		self.queue = cl.CommandQueue(self.ctx)
		self.kernel = cl.Program(self.ctx, self.kernel_code).build(cache_dir=self.cache_dir)

	def get_environment(self, atoms):
		if self.kernel is None:
			self.initialize()

		n_atoms = np.uint32(len(atoms))
		cell = np.array(atoms.cell, order='C')
		positions = np.array(atoms.positions, order='C')
		scaled_positions = np.zeros((n_atoms,4), dtype=np.float32, order='C')
		scaled_positions[:,0:3] = np.linalg.solve(cell.T, positions.T).T.astype(np.float32)

		neighborhood_idx = -np.ones((n_atoms,self.max_nbh), dtype=np.int32, order='C')
		offset = np.zeros((n_atoms,self.max_nbh), dtype=cl.cltypes.float3, order='C')

		b_scaled_positions = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=scaled_positions)
		b_neighborhood_idx = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=neighborhood_idx)
		b_offset = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=offset)

		cell = cell.astype(np.float32)
		self.kernel.neighbor_list_cl(self.queue, (n_atoms,), None, b_scaled_positions, b_offset, b_neighborhood_idx,
			n_atoms, self.max_nbh, self.cutoff,
			cell[0,0], cell[0,1], cell[0,2],
			cell[1,0], cell[1,1], cell[1,2],
			cell[2,0], cell[2,1], cell[2,2])

		e1 = cl.enqueue_copy(self.queue, offset, b_offset)
		e2 = cl.enqueue_copy(self.queue, neighborhood_idx, b_neighborhood_idx)

		e2.wait()
		tmp = neighborhood_idx.copy()
		tmp[tmp != -1] = 0
		I = int(self.max_nbh + np.max(np.sum(tmp, 1)))
		if I == 0:
			I = 1
		e1.wait()

		return neighborhood_idx[:,0:I], np.stack((offset[:,0:I]['x'], offset[:,0:I]['y'], offset[:,0:I]['z']), axis=2)
