import pyopencl as cl
import numpy as np

from schnetpack.environment import BaseEnvironmentProvider

class OpenCLEnvironmentProvider(BaseEnvironmentProvider):
	"""docstring for OpenCLEnvironmentProvider"""

	def __init__(self, cutoff, max_nbh=128):
		"""
		Args:
			cutoff (float): the cutoff inside which atoms are considered pairs
		"""
		self.cutoff = np.float32(cutoff)
		self.max_nbh = np.uint32(max_nbh)

		self.mf = cl.mem_flags
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)

		self.kernel = cl.Program(self.ctx, ''.join(open('neighbor_list_kernel.cl', 'r').readlines())).build()

	def get_environment(self, atoms):

		n_atoms = np.uint32(len(atoms))

		cell = np.array(atoms.get_cell(True), order='C').astype(np.float32)

		scaled_positions = np.zeros((n_atoms,4), dtype=np.float32, order='C')
		scaled_positions[:,0:3] = atoms.get_scaled_positions()

		neighborhood_idx = -np.ones((n_atoms,self.max_nbh), dtype=np.int32, order='C')
		offset = np.zeros((n_atoms,self.max_nbh), dtype=cl.cltypes.float3, order='C')

		s_buffer = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=scaled_positions)
		o_buffer = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=offset)
		nbh_idx_buffer = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=neighborhood_idx)
		

		self.kernel.neighbor_list_cl(self.queue, (n_atoms,), None, s_buffer, o_buffer, nbh_idx_buffer,
			n_atoms, self.max_nbh, self.cutoff,
			cell[0,0], cell[0,1], cell[0,2],
			cell[1,0], cell[1,1], cell[1,2],
			cell[2,0], cell[2,1], cell[2,2])

		e1 = cl.enqueue_copy(self.queue, offset, o_buffer)
		e2 = cl.enqueue_copy(self.queue, neighborhood_idx, nbh_idx_buffer)

		e2.wait()
		u_idx, n_nbh = np.unique(neighborhood_idx, return_counts=True)
		I = np.max(n_nbh[u_idx != -1])
		e1.wait()

		return neighborhood_idx[:,0:I], np.stack((offset[:,0:I]['x'], offset[:,0:I]['y'], offset[:,0:I]['z']), axis=2)
