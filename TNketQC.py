########################################################################
#
#
#                              TNketQC
#                            ===========
#
# A library containing functions for simulating quantum circuits in
# the pure-state (Schrodinger picture) with tensor-network and BP
#
# The main function here is apply_circuit, which applies a circuit on
# a TN in arbitrary graph, where each gate can be either 1-local or
# 2-local. The bond dimension is optionally compressed using BP and
# the Vidal gauge.
#
# History:
# ---------
#
# 5-Aug-2025  Itai  Initial version.
#
# 2-Oct-2025 Itai Various improvements/bugs corrected
#
# 20-Oct-2025 Itai Add Heisenberg evolution support:
#             1) Allow create_initial_T_list to create (Id)^{\otimes n}
#                initial product state (with mode='Id')
#             2) Added the gate_to_PTM function that maps a
#                1-local or 2-local unitary U to the PTM representation
#                of its Heisenberg evolution rho -> U rho U^\dagger
#
#             3) Added support for Heisenberg evolution in apply_circuit.
#                glist items that start with '@' are now interperted as
#                Heisenberg evolution gates. For example @CNOT01, etc...
#
# 24-Nov-2025: Add Single-Precision support: added a global variable
#              PRECISION_MODE, which can be either 'DP' (double-precision)
#              or 'SP' (single precision). This parameter can be set
#              via TNketQC_initialize()
#
# 18-Jan-2026: Added the function path_local_avs, which calculates the
#              expectation value of single-site operators along a
#              path.
#
# 3-Mar-2026:  Constantly normalize the enumerator & denominator in
#              path_local_avs in order to avoid a possible overflow
#
# 4-Mar-2026:  In apply_circuit(), added the calculation of the
#              total simulation fidelity as defined in arXiv:2503.20870v2.
#
#
# 14-Mar-2026: added the 'rXY_ZField' to get_gate
#
# 21-Mar-2026: 1) Renamed the function Heisenberg_unitary --> gate_to_PTM
#              2) Corrected a small bug in print_Vidal_gauge_statistics()
#              3) Add the function TPM_PEPS_to_PEPO()
#              4) Added the flag parameter normalize_PEPS to apply_circuit()
#
# 6-Apr-2026:  In apply_gate(), replace the call to 
#              BPSU.apply_2local_gate_notrunc() by a call to
#              BPSU.apply_gate_to_PEPS
#
# 6-Apr-2026:  Updated BPSU.lazy_PEPS_compression -> 
#              BPSU.lazy_PEPS_truncation
#              Updated also all the names of the Vigal Guage (VG)
#              functions in BPSU to their new names.
#
# 1-Jun-2026:  Added the function circuit_from_qasm_file and its 
#              supporting functions find_edge, parse_qasm_line
# 
# 1-Jun-2026:  Added support for CZ gate. 
#
# 1-Jun-2026:  In apply_circuit() function, update the BP_params['m_list']
#              entry with the BP messages each time BP is run
#
# 10-Jun-2026: In circuit_from_qasm_file, add support to 'barrier', 
#              'measure' gates (simply ignore them)
#
# 10-Jun-2026: In apply_circuit function, update the total (heuristic) 
#              truncation error fidelity to TN_parms: 
#              TN_params['total_err'], TN_params['total_f_sim']
#              Also change the printout text 'Step' -> 'Gate'
#
# 11-Jun-2026: Added support for rzz, rx, ry, h gates in 
#              circuit_from_qasm_file
#
# 11-Jun-2026: Allow circuit_from_qasm_file to create e_list, e_dict
#              on the fly.
#
# 12-Jun-2026: Added support for 'sxdg', 'szdg' gates in 
#              get_gate(), TNketQC_initialize() and support for 's', 
#              'sxdg', 'sdg' gates in circuit_from_qasm_file(). 
#              Also fix small regex bug in circuit_from_qasm_file().
#
# 29-Jun-2026: In get_gate function: change gate name 'Id' -> 'id', 
#              add gate 'h' as alias for 'H' and add gate 'rxx'.
#              
#
########################################################################


import numpy as np


import time
import psutil
import os
import re

from numpy import tensordot, zeros, conj, ones, array, exp, sin, cos, \
	sqrt, pi, dot, vdot, eye, log, trace, diag, sign

from numpy.linalg import norm
from scipy.linalg import expm

from qbp import calc_e_dict, qbp,  adj_vert

from BPSU import lazy_PEPS_truncation, apply_gate_to_PEPS, \
	VG_find_VG_from_BP, VG_merge_SU_weights, contract_leg

from TenQI import ID1, sigma_X, sigma_Y, sigma_Z, ketbra00, ketbra11, \
	op_to_mat


#
# ==================================================================
# The global precision: either 'DP' (double-precision) or 'SP'
# (single-precision)
#
# This parameter is set via TNketQC_initialize
#
# ==================================================================
#

PRECISION_MODE = 'DP'

#
# ------------------------  TNketQC_initialize  -------------------------
#

def TNketQC_initialize(mode='DP'):
	"""

	Set some global variables used as gates, as-well as the overall
	precision of the simulation.
	
	Note the global parameter PRECISION_MODE defines the global precision
	of the simulator (either DP or SP). It is used whenever new tensors
	are formed from scratch:
	(*) In the create_initial_T_list() function
	(*) In the get_gate() function

	Input Parameters:
	------------------

	mode --- The global precision of the simulation:
	         'SP' - Single precision - either float32 or complex64
	         'DP' - Double precision - either float64 or complex128
	         
	Output: None
	-------

	"""

	global CNOT01_gate, CNOT10_gate, H_gate, sZ_gate, sX_gate, \
		sZdg_gate, sXdg_gate, T_gate, Paulis1arr, Paulis2arr, CZ_gate,\
		PRECISION_MODE


	PRECISION_MODE=mode

	#
	# Some useful unitary gates
	#

	CNOT01_gate = tensordot(ketbra00, ID1, 0) +  tensordot(ketbra11, sigma_X, 0)
	CNOT10_gate = CNOT01_gate.transpose([2,3,0,1])
	
	CZ_gate = tensordot(ketbra00, ID1, 0) + tensordot(ketbra11, sigma_Z, 0)

	H_gate = array([[1,1],[1,-1]])/sqrt(2)

	sZ_gate = diag([1,1j])
	sX_gate = 0.5*array([ [1+1j, 1-1j], [1-1j, 1+1j]])
	
	sZdg_gate = conj(sZ_gate.T)
	sXdg_gate = conj(sX_gate.T)

	T_gate = diag([1, exp(1j*pi/4)])


	#
	# 1D list and 2D list of the Pauli matrices. This is used for
	# bookkeeping k=1 and k=2 channels in the PTM.
	#

	Paulis1 = [ID1, sigma_X, sigma_Y, sigma_Z]

	Paulis2 = [[None]*4 for i in range(4)]


	for i in range(4):
		for j in range(4):
			T = tensordot(Paulis1[i], Paulis1[j], 0)
			Paulis2[i][j] = op_to_mat(T)



	#
	# Create an array-version of Paulis lists
	#
	Paulis1arr = zeros([4,2,2], dtype=np.complex128)
	Paulis2arr = zeros([4,4,2,2,2,2], dtype=np.complex128)

	Paulis1arr[0,:,:] = ID1.copy()
	Paulis1arr[1,:,:] = sigma_X.copy()
	Paulis1arr[2,:,:] = sigma_Y.copy()
	Paulis1arr[3,:,:] = sigma_Z.copy()

	for al in range(4):
		for beta in range(4):
			Paulis2arr[al,beta,:,:,:,:] = tensordot(Paulis1[al],Paulis1[beta],0)



#
# ------------------------  process_memory  ----------------------------
#
def process_memory():
	r"""

	Measure the amount of (CPU) memory (in MB) used by the process

	"""

	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()
	mem_bytes = mem_info.rss

	mem_bytes = mem_bytes//2**20

	return mem_bytes




#
# ----------------------   create_initial_T_list   ---------------------
#

def create_initial_T_list(e_list, mode='|0>'):
	r"""

	Creates an initial state (stored in T_list) which is a product state.

	There are several modes, as decided by the input parameter mode:

	1. mode == '|0>': qubits prodcut state |0>^{\otimes n},
	2. mode == 'random-ket':  a product state of random 1-qubit ket states
	3. mode == 'Id': a mixed product state Id^{\otimes n} in the TPM

	In modes '|0>', 'random-ket' the physical dim is d=2. In 'Id' d=4.

	Input Parameters:
	-----------------
	e_list --- Holds the structure of the TN
	mode   --- Either '|0>' or 'random-ket' or 'Id'

	Output:
	-------
	T_list with bond-dim=1, describing the initial state.

	"""

	T_list = []

	if not mode in ['|0>', 'random-ket', 'Id']:
		print("\n")
		print("Error in create_initial_T_list() function:")
		print(f"mode = '{mode}' does not exist!")
		exit(1)

	for es in e_list:
		k = len(es)

		if mode == '|0>':
			T = array([1.0,0.0])

		if mode == 'random-ket':
			T = np.random.normal(size=[2])
			T = T/norm(T)

		if mode == 'Id':
			T = array([1.0, 0, 0, 0])

		d = T.shape[0]
		sh = [d] + [1]*k

		T = T.reshape(sh)

		#
		# Set the precision of the initial state, which determines the
		# precision of the tensors along the evolution.
		#
		if PRECISION_MODE=='SP':

			if T.dtype==np.complex128:
				T = T.astype(np.complex64)

			elif T.dtype==np.float64:
				T = T.astype(np.float32)


		T_list.append(T)



	return T_list

#
# -------------------------   get_gate   -------------------------------
#

def get_gate(gname, params=None):

	r"""

	Given a gate name (a string) and an optional gate parameters
	dictionary, return a tensor that encodes the corresponding gate.

	Currently, the following gates are supported:

	1) Single-qubit gates:
	   'Id', 'x', 'y', 'z', 'H', 'sx', 'sz', 'T', 'rx', 'ry', 'rz', 'u3'

	2) 2-qubits gates:
	   'CNOT01', 'CNOT10', 'r_AFH', 'rzz'


	Input Parameters:
	------------------
	gname  --- A string containing the name of the gate

	params --- A dictionary containing optional parameters for the gate


	Output:
	-------
	M --- A tensor with k input legs (dim=2) and k output legs that
	      encodes the gate in the shape (i_1,j_1; i_2,j_2; ...; i_k, j_k)

	      Currently only 1-qubit or 2-qubits gates are supported so k=1
	      or k=2.

	      As operator, the encoded gate is:
	      \sum_{i,j} M[i_1, j_1, ..., i_k, j_k] |i_1><j_1| \otimes ...
	        ... \otimes |i_k><j_k|


	"""


	match gname:

		case 'id':
			M = ID1

		case 'x':
			M = sigma_X

		case 'y':
			M = sigma_Y

		case 'z':
			M = sigma_Z

		case 'H':
			M = H_gate
			
		case 'h':
			M = H_gate

		case 'sx':
			M = sX_gate

		case 'sxdg':
			M = sXdg_gate

		case 'sz':
			M = sZ_gate
			
		case 'szdg':
			M = sZdg_gate

		case 'T':
			M = T_gate

		case 'rXY_ZField':
			#
			# rXY_ZField is a two-qubit rotation for an anti-ferromagnetic XY 
			# Hamiltonian with an external Z field. Essentially it is 
			# e^{-1j\cdot h}, where 
			# h := theta_XX*XX + theta_YY*YY + theta_Z_1*ZI+theta_Z_2*IZ
			#
			theta_XX = params['theta_XX']
			theta_YY = params['theta_YY']
			theta_Z_1 = params['theta_Z1']
			theta_Z_2 = params['theta_Z2']
			h = theta_XX * tensordot(sigma_X, sigma_X, 0) + \
					theta_YY * tensordot(sigma_Y, sigma_Y, 0) + \
					theta_Z_1 * tensordot(sigma_Z, ID1, 0) + \
					theta_Z_2 * tensordot(ID1, sigma_Z, 0)
			h_mat = op_to_mat(h)
			M_mat = expm(-1j * h_mat)
			M = mat_to_op(M_mat)

		case 'CNOT01':
			M = CNOT01_gate

		case 'CNOT10':
			M = CNOT10_gate
			
		case 'cz':
			M = CZ_gate
			
		case 'rx':
			theta = params['theta']
			M = expm(-0.5j*theta*sigma_X)

		case 'ry':
			theta = params['theta']
			M = expm(-0.5j*theta*sigma_Y)

		case 'rz':
			theta = params['theta']
			M = expm(-0.5j*theta*sigma_Z)

		case 'rzz':
			#
			# Rzz(theta) := exp(-0.5i*theta*ZZ)
			#
			theta = params['theta']

			M = cos(0.5*theta)*tensordot(ID1, ID1, 0) \
				- 1j*sin(0.5*theta)*tensordot(sigma_Z, sigma_Z, 0)

		case 'rxx':
			#
			# Rxx(theta) := exp(-0.5i*theta*XX)
			#
			theta = params['theta']

			M = cos(0.5*theta)*tensordot(ID1, ID1, 0) \
				- 1j*sin(0.5*theta)*tensordot(sigma_X, sigma_X, 0)



		case 'u3':
			#
			# u3 is an arbitrary 1-qubit rotation
			#
			theta0 = params['theta0']
			theta1 = params['theta1']
			theta2 = params['theta2']

			M = array([ [cos(theta0/2), -exp(1j*theta2)*sin(theta0/2)], \
				[exp(1j*theta1)*sin(theta0/2), exp(1j*(theta1+theta2))*cos(theta0/2)]])

		case 'r_AFH':
			#
			# r_AFH is an 'anti-ferromagnetic Heisenberg' Trotter-Suzuki
			# rotation. Essentially it is e^{-1j\cdot h}, where
			#
			# h := theta_XX*XX + theta_YY*YY + theta_ZZ*ZZ
			#
			theta_XX = params['theta_XX']
			theta_YY = params['theta_YY']
			theta_ZZ = params['theta_ZZ']

			h = theta_XX*tensordot(sigma_X, sigma_X,0) + \
				theta_YY*tensordot(sigma_Y, sigma_Y,0) + \
				theta_ZZ*tensordot(sigma_Z, sigma_Z,0)

			h_mat = op_to_mat(h)

			M_mat = expm(-1j*h_mat)

			M = mat_to_op(M_mat)


		case _:
			print(f"Error in get_gate() function! gate {gname} is undefined")
			exit(1)

	#
	# Set the precision of the final gate; reduce to single-precision
	# if necessary
	#
	if PRECISION_MODE=='SP':
		if M.dtype==np.complex128:
			M = M.astype(np.complex64)
		elif M.dtype==np.float64:
			M = M.astype(np.float32)


	return M

#
# ------------------------  apply_gate  ------------------------------
#
def apply_gate(TN_params, M, i=None, e=None):
	r"""

	Applies a gate M to the TN *without* compression.

	The gate can be either 1-qubit (in which case the vertex i is given),
	or it can be 2-qubit gate (in which case the edge e is given)

	The 2-local gate is applied via the BPSU function
	apply_2local_gate_notrunc(). In such case, e=(v1, v2) with v1<v2,
	and M assumed to be given given as [i1,j1; i2,j2]

	Input Parameters:
	------------------

	TN_params --- Holds the TN parameters: T_list, e_list, e_dict

	M         --- The gate to apply. Either 1-local or 2-local

	i         --- The qubit index (when M is 1-local)

	e         --- The edge label (when M is 2-local)


	Output:
	-------

	T_list --- The updated T_list

	"""

	if (i is None) and (e is None):

		print("Error in apply_gate: both i and e cannot be None!")
		exit(1)

	if i is not None and e is not None:
		print("Error in apply_gate: i and e cannot be both not None!")
		exit(1)

	T_list = TN_params['T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']

	T_list = apply_gate_to_PEPS(T_list, e_list,  e_dict, M, i, e)

	return T_list


#
# -------------------------  gate_to_PTM  -----------------------
#

def gate_to_PTM(U):

	r"""

	Given a unitary U of a gate (either 1-qubit of 2-qubits), produces the
	tensor that represents the channel U\cdot U^\dagger in the 
	PTM representation.


	Input Parameters:
	-------------------
	U --- The unitary of the gate.
	      U can be either 1 local or 2 local. In the 2-local case, it is
	      given as a tensor in the form [i0,j0; i1,j1] where

	      U = \sum_{i0,j0; i1,j1} U[i0,j0; i1,j1] \cdot |i0><j0| \otimes |i1><j1|


	Output:
	-------

	The channel in PTM representation:

	(*)  for 1-local: T_{bet,al}
	(*)  for 2-local: T_{bet0,al0; bet1,al1}

	Explicitly, for the 2-local case, for

	rho = sum_{al0,al1} rho_{al0,al1} sigma_al0\otimes \sigma_al1

	We have:

	E(rho) = sum_{bet0,bet1} rho'_{bet0,bet1} sigma_bet0\otimes \sigma_bet1

	where:

	rho'_{bet0,bet1} := \sum_{al0,al1} T_{bet0,al0; bet1,al1} rho_{al0,al1}



	"""


	k = len(U.shape)//2

	if k==1:
		#
		# It's a 1-qubit gate. Resulting tensor is T_{alpha,beta}
		#

		P_alpha_U = tensordot(Paulis1arr, U, axes=([2],[0]))
		P_beta_Udag =  tensordot(Paulis1arr, conj(U.T), axes=([2],[0]))

		# our tensors are of the form [alpha, i, j]

		T = (tensordot(P_alpha_U, P_beta_Udag, axes=([1,2],[2,1]))).real

	if k==2:
		#
		# It's a 2-qubits gate.
		#
		# Tensor should be of the form [alpha0,beta0; alpha1,beta1]
		#


		#
		# Recall, Paulis2arr is of the form:
		# [al0,al1; i0,j0; i1,j1]
		#
		# U is of the form [i0,j0; i1,j1]
		#

		P_alpha_U = tensordot(Paulis2arr, U, axes=([3,5],[0,2]))
		P_beta_Udag = tensordot(Paulis2arr, conj(U), axes=([3,5],[1,3]))

		#                                0   1      2  3  4 5
		# Now P_alpha_U is of the form [al0,al10; i0,i1,j0,j1]
		# P_beta_Udag form:            [beta0,beta1;

		T = (tensordot(P_alpha_U, P_beta_Udag, axes=([2,3,4,5],[4,5,2,3]))).real
		#
		# T form: [al0,al1,beta0,beta1]
		#
		T = T.transpose([0,2,1,3])


	#
	# Normalize due to the Pauli inner-product
	#
	T = T/2**k

	return T


#
# -------------------------  parse_qasm_line  --------------------------
#
def parse_qasm_line(line):
	
	r"""
	
	Reads a QASM line, and extract from it:
	1. gate name
	2. qubits acted on
	3. Rotation angle (if exists)
	
	When a rotation angle is specified, the function knows how to translate
	expressions containing pi to their actual real value.
	
	NOTE: this function was written by AI Claude
	
	Input Parameters:
	-----------------
	line --- a string containing the QASM line
	
	Output:
	-------
	gname  --- The QASM gate name (i.e., 'cx', 'sz', etc)
	
	qubits --- A list of the qubits on which the gates act
	
	param  --- The rotation angle in the case of a rotation
	
	"""
	
	
	line = line.strip()
	if not line or line.startswith('//'):
			return None, None, None

	# Match gate with optional parameter and one or two qubits
	# e.g. "cz q[6],q[5]" or "rz(pi/2) q[4]" or "sx q[2]"
	pattern = r'(\w+)(?:\(([^)]*)\))?\s+q\[(\d+)\](?:,\s*q\[(\d+)\])?'
	m = re.match(pattern, line)
	if not m:
			return None, None, None

	gate = m.group(1)
	param_str = m.group(2)
	q1 = int(m.group(3))
	q2 = int(m.group(4)) if m.group(4) is not None else None

	qubits = [q1] if q2 is None else [q1, q2]

	# Evaluate parameter expression safely
	param = None
	if param_str is not None:
			expr = param_str.strip()
			expr = expr.replace('pi', str(np.pi))
			param = eval(expr)

	return (gate, qubits, param)

#
# ------------------------   find_edge    ------------------------------
#

def find_edge(i, j, e_list, e_dict):
	
	r"""
	
	Find the edge that connects the i,j vertices. If no such edge exists,
	return None
	
	
	"""
	
	es = e_list[i]
	for e in es:
		if adj_vert(i, e, e_dict)==j:
			return e
			
	return None
	

#
# -------------------   circuit_from_qasm_file   -----------------------
#

def circuit_from_qasm_file(fname, e_list=None, e_dict=None):
	r"""
	
	Reads a QASM file and translates it into a circuit.
	
	In a QASM file, every line corresponds to a gate, which can be either
	1-local or 2-local. Typical lines may look like:
	
	rz(5*pi/4) q[6];
	cz q[5],q[6];
	
	Any gate acting on qubit q[i] will be mapped to a gate acting on
	vertex i in the TN. Therefore, it is expected that:
	
	1. Any qubit q[i] should be smaller than len(e_list)
	2. If a 2-qubits gate acts on q[i],q[j], then the TN has an edge e=(i,j).
	
	The function has two running modes:
	(*) If e_list is given, then we check that every gate correspond
	    to a existent qubits and edges.
	(*) If e_list is not given, then e_list and e_dict are constructed
	    on the fly.
	
	
	Input Parameters:
	------------------
	fname          --- The QASM file name
	e_list, e_dict --- TN structure. If not given then they are constructed
	                   on the fly
	
	Output:
	-------
	Either:
		g_list --- The gates list of the circuit
	Or 
		g_list, e_list, e_dict  (when e_list is not given as input)
	
	
	"""
	
	MAX_QUBITS = 10000
	
	
	#
	# See if e_list is given. If not then set construct_e_list=True
	# and define a list all_es of all possible edges
	#
	if e_list is None:
		e_list = [[] for i in range(MAX_QUBITS)]
		construct_e_list = True
		max_q = 0
		all_es = []
		
	else:
		construct_e_list = False
		if e_dict is None:
			e_dict = calc_e_dict(e_list)
	
		n = len(e_list)
	
	#
	# Open the QASM file and read its lines to a list Lines
	#
	fin = open(fname, 'r')
	Lines = fin.readlines()
	fin.close()
	
	glist = []
	
	#
	# Now go over the lines and translate each QASM gate to our gate
	#
	for i, L in enumerate(Lines):

		(gname, qubits, param) = parse_qasm_line(L)
		
		if gname is None or qubits is None or gname in ['qreg', 'barrier', 'measure']:
			continue
			
			
		if len(qubits)==1:
			#
			# Its a 1-qubit gate
			#
			q1 = qubits[0]
			
			if construct_e_list:
				#
				# If we're constructing e_list of the fly then see if we need
				# to increase the total number of qubits
				#
				if q1>max_q:
					max_q = q1
			else:
				#
				# Otherwise, just check that it is an existent qubit
				#
				if q1>=n:
					print("\n")
					print("Error in circuit_from_qasm_file() function:")
					print(f"1-local gate '{L}' in line {i} corresponds to a "\
						f"non-existing qubit index {q1} "\
						f"(there are at most {n} qubits in the TN)!")
					exit(1)
				
			e = None
		else:
			#
			# Its a 2-qubits gate. Find the edge that connects q1,q2
			#
			
			q1,q2 = qubits[0],qubits[1]
			
			#
			# Make sure q1<q2
			#
			if q1>q2:
				q1,q2 = q2,q1
			
			
			if construct_e_list:
				#
				# If we're constructing e_list on the fly then create the 
				# corresponding e, and see if we need to add it to e_list
				# (if its the first time we encounter it)
				#
				e = f'e{q1}-{q2}'
				if not e in all_es:
					all_es.append(e)
					e_list[q1].append(e)
					e_list[q2].append(e)
					
				if q1>max_q:
					max_q = q1
					
				if q2>max_q:
					max_q = q2

					
			else:
				e = find_edge(q1, q2, e_list, e_dict)
				
				if e is None:
					print("\n")
					print("Error in circuit_from_qasm_file() function:")
					print(f"2-local gate '{L}' in line {i} corresponds to a "\
						f"non-existing edge {e}!")
					exit(1)
				
		
		match gname:

			case 'cx':
				#
				# We apply C-NOT where qubits[0] is the control and qubits[1] 
				# is the target. 
				#
				if qubits[0]==q1:
					g = ('CNOT01', None, e, None)
				else:
					g = ('CNOT10', None, e, None)
			
			case 'cz':
				g = ('cz', None, e, None)
				
			case 'rzz':
				g = ('rzz', None, e, {'theta':param})
				
			case 'sx':
				g = ('sx', q1, None, None)
				
			case 'sxdg':
				g = ('sxdg', q1, None, None)

			case 's':
				g = ('sz', q1, None, None)
				
			case 'sdg':
				g = ('szdg', q1, None, None)

				
			case 'rz':
				g = ('rz', q1, None, {'theta':param})
				
			case 'rx':
				g = ('rx', q1, None, {'theta':param})
				
			case 'ry':
				g = ('ry', q1, None, {'theta':param})

			case 'x':
				g = ('x', q1, None, None)
				
			case 'h':
				g = ('H', q1, None, None)
				
			case 'y':
				g = ('y', q1, None, None)
				
			case 'z':
				g = ('z', q1, None, None)
				


			case _:
				print("\n")
				print("Error in circuit_from_qasm_file() function:")
				print(f"The gate {gname} in '{L}' in line {i} is not "\
					"implemented yet in the function!")
				exit(1)
				
		glist.append(g)
	
	
	if construct_e_list:
		#
		# If we're constructing e_list on the fly then truncate it up
		# to the maximal qubit used, and calculate e_dict
		#
		e_list= e_list[:(max_q+1)]
		e_dict = calc_e_dict(e_list)
		
		return glist, e_list, e_dict
		
	else:
		return glist
		


#
# --------------------   print_Vidal_gauge_statistics   ----------------
#
def print_Vidal_gauge_statistics(w_dict):
	"""

	Given a Vidal gauge weight dictionary w_dict, calculate the entropy
	at every edge and print a global statistics of it:
	(*) Average edge entropy
	(*) Min edge entropy
	(*) Max edge entropy

	w_dict is a dictionary of the form {e:w}, where e is the edge label
	and w is a numpy array with the Vidal gauge weights.


	"""

	S = 0
	S2 = 0
	n=0

	smin = 1e9
	smax = 0

	for e in w_dict:
		w = w_dict[e]
		p = abs(w**2)
		p = p/sum(p)
		log_p = log(p)/log(2)
		entropy = -sum(p*log_p)

		if entropy > smax:
			smax = entropy
		if entropy < smin:
			smin = entropy

		S += entropy
		S2 += entropy**2
		n +=1

	avS = S/n
	avS2 = S2/n
	dS = sqrt(abs(avS2 - avS**2))

	print()
	print(f"Vidal edge entropy:  <S> = {avS:.6g} +/- {dS:.6g} "\
			f"min(S)={smin:.6g}   max(S)={smax:.6g}")
	print( "-------------------")
	print()


#
# -------------------------    calc_RDM1    ----------------------------
#
def calc_RDM1(v, T_list, e_list, e_dict, m_list):
	"""

	Given a ket TN and a set of converged BP messages, use these messages
	to approximate the 1-qubit RDM of a vertex v.

	Input Parameters:
	------------------
	v --- The vertex index

	T_list, e_list, e_dict --- TN and its structure

	m_list --- The converged BP messages

	Output:
	-------
	rho1 --- The 2x2 matrix of the RDM at qubit v

	"""

	es = e_list[v]
	T = T_list[v]
	k = len(es)

	#
	# First contract the ket leg of the incoming messages to the ket
	# legs of T
	#
	Tket = T.copy()
	for l, e in enumerate(es):
		j = adj_vert(v, e, e_dict)
		m = m_list[j][v]
		Tket = contract_leg(Tket, m, l)

	#
	# Now contract the bra of T to the previous tensor
	#
	sh = list(range(1,k+1))
	rho1 = tensordot(Tket, conj(T) ,axes=(sh, sh))

	#
	# Normalize the resultant rho1
	#
	rho1 = rho1/trace(rho1)

	return rho1


#
# -------------------------    calc_RDM2    ----------------------------
#
def calc_RDM2(e, T_list, e_list, e_dict, m_list):

	"""

	Given a ket TN and a converged set of BP messages, calculate the
	2-local RDM of a pair of neighboring qubits on an edge e.

	Input Parameters:
	-----------------
	e --- The label of the edge for which we want the 2-local RDM

	T_list, e_list, e_dict --- The TN and its structure

	m_list --- List of converged BP messages


	Output:
	--------
	rho2 --- The resultant 2-local RDM. rho2 is given as a degree 4 array.
	         if e=(a,b) then the indices of rho2 are
	         [a_bra,a_ket; b_bra, b_ket]



	"""

	v1,leg1, v2,leg2 = e_dict[e]

	T1 = T_list[v1]
	T2 = T_list[v2]

	D = T1.shape[leg1+1]  # Original dimension of the common leg

	#
	# Contract the incoming messages to T1, T2 (except for the message
	# of the common leg)
	#

	es1 = e_list[v1]
	for leg,f in enumerate(es1):

		if f==e:
			continue

		j = adj_vert(v1, f, e_dict)
		m = m_list[j][v1]

		T1 = contract_leg(T1, m, leg)


	#
	# contract T1 with the bra along all legs except for the physical
	# leg and the joint leg with T2
	#
	L = len(T1.shape)
	sh = list(range(L))
	sh.remove(0)
	sh.remove(leg1+1)

	T1ketbra = tensordot(T1, conj(T_list[v1]), axes=(sh, sh))
	# T1ketbra form: d, D, d*, D*


	es2 = e_list[v2]
	for leg,f in enumerate(es2):

		if f==e:
			continue

		j = adj_vert(v2, f, e_dict)
		m = m_list[j][v2]

		T2 = contract_leg(T2, m, leg)


	#
	# contract T2 with the bra along all legs except for the physical
	# leg and the joint leg with T1
	#
	L = len(T2.shape)
	sh = list(range(L))
	sh.remove(0)
	sh.remove(leg2+1)

	T2ketbra = tensordot(T2, conj(T_list[v2]), axes=(sh, sh))
	# T2ketbra form: d, D, d*, D*


	#
	# get rho12 by contracting T1ketbra with T2ketbra along D,D*
	#

	rho12 = tensordot(T1ketbra, T2ketbra, axes=([1,3],[1,3]))

	#
	# Calculate the trace and normalize rho2
	#
	tr = trace(rho12, axis1=0, axis2=1)
	tr = trace(tr, axis1=0, axis2=1)

	rho12 = rho12/tr

	rho12 = rho12

	return rho12




#
# ------------------------   path_local_avs  ---------------------------
#
def path_local_avs(T_list, e_list, e_dict, m_list, v_list, op_list):
	r"""

	Given a linear path of vertices on the graph q0,q1, ... and a
	corresponding list of 1-site operators op0, op1, ...
	uses the converged BP messages to calculate the expectation
	value  <op0 \otimes op1 \otimes ... >

	Note:
	------

	q0,q1,q2, ... must be a valid linear path on the graph. So q_i
	must be a neighbor of q_{i+1}. Also we assume it to be a simple
	path a not a loop, or not a path that intersects itself.

	Input Parameters:
	------------------

	T_list, e_list, e_dict --- TN info

	m_list  --- converged BP messages

	v_list  --- The list [q0,q1,q2, ...] of the vertices that make up the
	            path

	op_list --- The list [op0, op1, op2, ...] of the corresponding 1-site
	            operators


	Output:
	-------

	The *normalized* expectation value






	"""


	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~   out_m
	#
	def out_msg(Tket, Tbra, in_m_list, in_legs_list):

		r"""
		  Given ket & bra tensors, together with a list of incoming
		  messages and their corresponding legs, contract the messages
		  to the ket-bra legs and return a tensor made of the remaining
		  uncontracted indices.

		"""

		# ax will become the list of indices we contract over when we
		# contract the ket & bra

		ax = [0]

		#
		# Contract the incoming messages to the ket
		#
		for i, leg in enumerate(in_legs_list):
			Tket = contract_leg(Tket, in_m_list[i], leg)
			ax.append(leg+1)

		#
		# Contract the ket & bra to get the final output tensor
		#
		out_m = tensordot(Tket, Tbra, axes=(ax,ax))

		return out_m


	#
	# ~~~~~~~~~~~~~~~~~~~~~~~  function starts here  ~~~~~~~~~~~~~~~~~~~~~
	#
	path_n = len(v_list)

	prev_leg = None
	prev_v = None

	out_m    = None
	out_m_nr = None

	#
	# Go over the path. Contract the BP messages together with the
	# tensor we got from the previous site.
	#
	# Note: we perform two contractions in parallel: one for the operators
	#       expectation value and one with ID for the normalization.
	#
	for i in range(path_n):
		v = v_list[i]

		in_m_list = []
		in_m_nr_list = []

		in_legs_list = []

		if i<path_n-1:
			next_v = v_list[i+1]
		else:
			next_v = None

		found_next_vert=False

		#
		# Go over all the legs of the tensor and either match it with
		# the incoming BP message, or the tensor from the previous step,
		# or leave it open if it is pointing to the next vertex on the path.
		#
		for leg, e in enumerate(e_list[v]):

			j = adj_vert(v, e, e_dict)

			if j==prev_v:
				in_m_list.append(out_m)
				in_m_nr_list.append(out_m_nr)
				in_legs_list.append(leg)

			elif j != next_v:
				in_m_list.append(m_list[j][v])
				in_m_nr_list.append(m_list[j][v])
				in_legs_list.append(leg)
			else:
				found_next_vert = True

		#
		# Perfrom a simple test to verify that we're on a connected path
		#
		if i<path_n-1 and not found_next_vert:
			print("Error in path_local_avs:")
			print(f"path is: v_list={v_list} but {next_v} is not a "\
				f"neighbor of {v} !")
			exit(1)

		Tket = tensordot(op_list[i], T_list[v], axes=([1],[0]))
		Tbra = conj(T_list[v])
		out_m = out_msg(Tket, Tbra, in_m_list, in_legs_list)

		# The normalization tensor
		out_m_nr = out_msg(T_list[v], Tbra, in_m_nr_list, in_legs_list)

		#
		# Normalize both the enumerator and denominator (with the *same*
		# constant) in order to avoid possible overflow
		#
		nr = norm(out_m_nr)
		out_m_nr = out_m_nr/nr
		out_m = out_m/nr

		prev_v = v


	av = out_m/out_m_nr

	return av



#
# ------------------------   TN_size  -------------------------------
#

def TN_size(T_list):

	r"""

	Calculates the total number of *bytes* that the TN occupies and a 
	string with the general bond-dim shape of each tensor.

	"""

	TN_sz = 0
	TN_str = ''

	for i,T in enumerate(T_list):
		TN_sz += T.nbytes
		TN_str = TN_str + f'T[{i}]={T.shape}; '

	return TN_sz, TN_str





#
# ------------------------   TPM_PEPS_to_PEPO   -------------------------
#

def TPM_PEPS_to_PEPO(T_list):
	r"""
	
	Given a PEPS that encodes a multiqubit operator in the 
	Pauli-Transfer-Matrix (PTM) method, convert it to a PEPO operator.
	
	This is a local transformation, where every local tensor changes as:
	
	[4, D_1, D_2, ...]  ===>  [2,2, D_1, D_2, ...]
	
	where the [2,2] legs in the new tensors are now the ket-bra physical
	legs of the PEPO.
	
	"""
	
	T2_list = []
	n = len(T_list)
	
	for i in range(n):
		T = T_list[i]
		
		T2 = tensordot(Paulis1arr, T, axes=([0],[0]))
		
		T2_list.append(T2)
		
	return T2_list
		
	
	

#
# ----------------------- apply_circuit  ----------------------------
#

def apply_circuit(TN_params, BP_params, glist, normalize_PEPS=True):

	r"""

	Applies a quantum circuit to an TN ket state.

	The quantum circuit is given by a list glist. Each element of the
	list is a 4-tuple the form

	glist = [(gname, i, e, parmas), (gname, i, e, params), ...]

	Each such element indicates either:
	(1) A gate to apply
	(2) A global action to do on the TN

	In the first case, gname is a string holding the gate name
	(from get_gate() func), i or e specify the location:
	  (*) i for 1-qubit gate,
	  (*) e for 2-qubit gate

	params is an optional dictionary of parameters for the gates (can
	be also None).

	  Hiesenberg Evolution:
	  ---------------------
	  If the gate name is preceded with a '@', then it is interpreted
	  as a Heisenberg evolution operator in the PTM representaiton.
	  In other words, if U is the gate, then the actual linear map
	  that is applied is rho -> U \rho U^\dagger

	In the second case, if gname starts with a '*', it specifies a global
	action. In such case i=e=None and params is an optional parameter. 
	Currently, the following global actions are supported:

	1) '*compress':
	   Perform a lazy_L2 compression of the TN according to the
	   compression parameters D_max, L2thresh that are specified in
	   TN_params. This is done by first running qbp, and then sending the
	   converged BP messages to the function BPSU.lazy_PEPS_truncation()
	   that uses these messages to perform the (lossy) compression.

	2) '*message':
	   Display a message (stored in params)

	3) '*BP':
	   Run BP and store the messages.

	4) '*vgauge':
	   Run BP and use the BP messages to pass to the Vidal gauge on the
	   TN

	5) '*vgauge-noBP':
	   Like *vgauge, but do not run BP; instead, use last BP messages.
	   This is useful for cases where we want to use the BP messages also
	   for other tasks. So we first run BP, use the messages to whatever
	   we want and then we can also use them to pass to Vidal gauge without
	   running BP twice.

	6) '*ext-func':
	   Call an external function specified by params. The function will
	   must be of the form

	   ext_func(TN_params, BP_params, glist, t, func_params)

	   where func_params are also given inside params.



	Input Parameters:
	------------------

	TN_params --- The TN params (T_list, e_list, etc)

	BP_params --- The BP parameters used in the compression

	glist     --- The list of the gates in the circuit.
	
	normalize_PEPS --- Whether or not to normalize the PEPS tensors
	                   after each iteration. This is useful when using
	                   the function for back-operator-propagation (BOP), 
	                   in which we do not want to change the normalization
	                   of the operator that is encoded by the TN.

	Output:
	-------

	TN_list --- The resultant state.

	"""

	T_list = TN_params['T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']

	D_max = TN_params['D_max']
	L2thresh = TN_params['L2thresh']

	BP_max_iter = BP_params['BP_max_iter']
	BP_delta    = BP_params['BP_delta']
	BP_damping  = BP_params['BP_damping']

	m_list = None
	BP_iter_no = None
	BP_err = None

	total_err = 0
	total_f_sim = 1

	TN_params['total_err']   = total_err
	TN_params['total_f_sim'] = total_f_sim


	T = len(glist)
	


	for t, gate in enumerate(glist):

		gname, i, e, params = gate

		display_comp_msg = False

		match gname:

			case '*compress':

				#
				# ^^^^^^^^^^^^^^^^^^^^^   *compress    ^^^^^^^^^^^^^^^^^^^^^^^^^
				#

				print()
				print("            ---- Compressing ----")

				mem = process_memory()
				print()
				print(f"> Memory used before compression: {mem} MB\n")

				t0 = time.time()

				#
				# Perform lazy BP compression. We first run qbp, and then
				# pass the converged messages to lazy_PEPS_truncation
				#
				
				print()
				print(f"Running BP for compression...")
				print(f"BP params:  delta={BP_delta:.6g}  "\
					f"max_iter={BP_max_iter}  damping={BP_damping:.6g}\n")
				

				m_list, BP_err, iter_no = qbp(T_list, e_list, e_dict, initial_m='U', \
					max_iter=BP_max_iter, delta=BP_delta, damping=BP_damping)
					
				print()
				print(f"> BP ended after {iter_no} iterations with "\
					f"BP-err={BP_err:.6g}\n")
				
				BP_params['m_list'] = m_list
				BP_params['BP_err'] = BP_err
				BP_params['BP_iter_no'] = BP_iter_no
				
				
				print()
				print(f"Using the converged BP messages to compress the TN...")
				T_list, err, f_sim = lazy_PEPS_truncation(T_list, e_list, e_dict, \
					m_list, Dmax=D_max, L2thresh=L2thresh, \
					normalize_tensors=normalize_PEPS)

				mem = process_memory()
				print()
				print("> Compression done.")
				print(f"> Simulation memory used after compression: {mem} MB\n")


				t1 = time.time()

				TN_sz, TN_str = TN_size(T_list)
				print("Resultant TN shape: ")
				print(TN_str)
				print()
				print(f"TN size: {TN_sz//2**20} MB     "\
					f"Compression done in {(t1-t0):.6g} secs.\n")

				total_err += err
				total_f_sim *= f_sim
				
				display_comp_msg = True

			case '*message':
				#
				# ^^^^^^^^^^^^^^^^^^^^^^   *message    ^^^^^^^^^^^^^^^^^^^^^^^^^
				#
				msg_str = params['msg_str']

				print()
				print(msg_str)
				print()

			case '*vgauge' | '*vgauge-noBP':
				#
				# ^^^^^^^^^^^^^^^^^^^^^^   *vagauge    ^^^^^^^^^^^^^^^^^^^^^^^^^
				#


				t0 = time.time()
				if gname=='*vgauge':
					print()
					print("            ----  Moving to Vidal gauge  ----\n")

					print("Running BP for the Vidal gauge...")

					m_list, BP_err, BP_iter_no = qbp(T_list, e_list, e_dict, \
						initial_m='U', max_iter=BP_max_iter, delta=BP_delta, \
						damping=BP_damping)

					print(f"... Vidal BP ended after {BP_iter_no} iterations "\
						f"with BP-err={BP_err:.6g}\n")
						
					BP_params['m_list'] = m_list
					BP_params['BP_err'] = BP_err
					BP_params['BP_iter_no'] = BP_iter_no
						

				else:
					print()
					print("        ----  Moving to Vidal gauge  (no BP)  ----\n")

				T_list, w_dict = VG_find_VG_from_BP(T_list, e_dict, m_list)

				T_list = VG_merge_SU_weights(T_list, e_dict, w_dict)


				t1 = time.time()

				print(f"... Done in {(t1-t0):.6g} secs.")

				if params is not None:
					if params['stat']:
						print_Vidal_gauge_statistics(w_dict)

			case '*BP':
				#
				# ^^^^^^^^^^^^^^^^^^^^^^^^^   *BP    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
				#

				t0 = time.time()
				print()
				print("            ----  Running BP  ----\n")

				m_list, BP_err, BP_iter_no = qbp(T_list, e_list, e_dict, initial_m='U', \
					max_iter=BP_max_iter, delta=BP_delta, damping=BP_damping)

				t1 = time.time()
				print(f"... BP ended after {BP_iter_no} iterations with "\
					f"BP-err={BP_err:.6g}\n")

				print(f"... Done in {(t1-t0):.6g} secs.")

				BP_params['m_list'] = m_list
				BP_params['BP_err'] = BP_err
				BP_params['BP_iter_no'] = BP_iter_no


			case '*ext-func':
				#
				# ^^^^^^^^^^^^^^^^^^^^^   *ext-func    ^^^^^^^^^^^^^^^^^^^^^^^^^
				#


				print()
				print("            ----  Calling external function  ----\n")


				ext_func = params['ext_func']
				func_params = params['func_params']

				T_list, err = ext_func(TN_params, BP_params, glist, t, func_params)

				total_err += err

			case other:
				#
				# ^^^^^^^^^^^^^^^^^^^^^   Ordinary Gate    ^^^^^^^^^^^^^^^^^^^^^
				#

				#
				# So its a gate
				#

				if e is None:
					gstr=f"qubit {i}"
				else:
					i1,i_leg, j1,j_leg = e_dict[e]
					gstr=f"qubits ({i1},{j1}) (e={e})"



				gate_str = gname

				if type(params) is dict:
					if 'theta' in params:
						gate_str = f"{gname}({params['theta']:.6g})"

				if gname[0]=='@':
					#
					# Its a Hiesenberg mode gate. So we need to turn the regular
					# unitary U into a matrix M in the PTM representation that
					# encodes the transformation
					#
					#                rho --> U \rho U^\dagger
					#

					U = get_gate(gname[1:], params)

					M = gate_to_PTM(U)


				else:
					M = get_gate(gname, params)

				#
				# Apply the gate to the TN
				#
				print()
				print(f"Gate [{t}/{T}]: applying {gate_str} on {gstr}")

				T_list = apply_gate(TN_params, M, i, e)



		if display_comp_msg:
			print(f"<<< local-compression-err={err:.6g}, local-fid={f_sim:.6g}, "\
				f"total-err={total_err:.6g}, total_simulation_fid={total_f_sim:.6g} >>>")
				
		#
		# Update the TN and BP params
		#

		TN_params['T_list']      = T_list
		TN_params['total_err']   = total_err
		TN_params['total_f_sim'] = total_f_sim

		BP_params['m_list'] = m_list
		BP_params['BP_err'] = BP_err
		BP_params['BP_iter_no'] = BP_iter_no


	return T_list










