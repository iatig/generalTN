#!/usr/bin/python3

########################################################################
#
#
#                        heavyhex_sim.py
#                        ===============
#
# Use PEPS-BP to simulate a "Trotter" circuit on an IBM-like heavyhex 
# grid
#
# History:
# --------
#
# 10-Jun-2026  Itai  Initial version
#
########################################################################


from os import environ

#
# Set the number of CPU threads used
#
N_THREADS = '4'

environ["OMP_NUM_THREADS"] = N_THREADS  # export OMP_NUM_THREADS=4
environ["OPENBLAS_NUM_THREADS"] = N_THREADS  # export OPENBLAS_NUM_THREADS=4
environ["MKL_NUM_THREADS"] = N_THREADS  # export MKL_NUM_THREADS=6
environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS  # export VECLIB_MAXIMUM_THREADS=4
environ["NUMEXPR_NUM_THREADS"] = N_THREADS  # export NUMEXPR_NUM_THREADS=6

import sys
import time

import numpy as np

import pickle
import time

from qbp import adj_vert, calc_e_dict, qbp, get_Bethe_free_energy

from ibm_grid import create_ibm_grid

from TNketQC import TNketQC_initialize, create_initial_T_list, \
	apply_circuit, calc_RDM1, calc_RDM2

from numpy import array, exp, sqrt, pi, trace


#
# ---------------------- local_avs -----------------------------
#

def local_avs(TN_params, BP_params, glist, t, func_params):
	"""

	This function is called whenever we want to calculate local
	expectation values.
	
	It calculates the local <Z_i> averages for all the qubits. This
	is done by using the current BP messages to calculate the local 1-RDM
	and from that calculate <Z_i> := Tr(rho_i Z_i)

	"""

	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~   calc_Z
	#
	def calc_Z(i):
		sigma_Z = array([[1, 0], [0, -1]])

		rho_i = calc_RDM1(i, T_list, e_list, e_dict, m_list)

		avZ = trace(rho_i @ sigma_Z)

		return avZ.real

	#
	# ~~~~~~~~~~~~~~~~~~~~~~~  function starts here  ~~~~~~~~~~~~~~~~~~~~~
	#

	T_list = TN_params['T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']
	m_list = BP_params['m_list']

	step = func_params  # which step we're in

	for i in range(len(e_list)):
		Z = calc_Z(i)
		print(f"Averages: <Z{i}>({step}) = {Z}")
	print()

	return T_list, 0


#
# ----------------------  one_Ising_Trotter_step  ----------------------
#
def one_Ising_Trotter_step(e_list, e_dict, layers, \
						   theta_x, theta_z, theta_zz, step, D_max=0):
	"""

	Creates a list with the gates of a single "Trotter step".
	
	The single Trotter step is decomposed into L layers. In each layer
	we act with:
	1. Rx(theta_x/L) --> Rz(theta_z/L) on all qubits i
	2. Rzz(theta_z) on all (i,j) pairs in the layer
	
	When the potential bond dimension (i.e, D=2^steps) exceeds D_max, 
	we call the compression after each layer.
	
	After the last layer is applied, we call BP and use the converged 
	BP messages for:
	1. Calculating the local <Z_i> expectation values
	2. Re-gauge the TN according to the Vidal Gauge for sake of numerical
	   stability.
	
	Input Parameters:
	------------------
	e_list, e_dict:
	  TN structure
	
	layers:
	  A list of layers. Each layers is the list of edges that define it.
	           
	theta_x, theta_z, theta_zz:
	  The rotations of the Trotter circuit
	
	step:
	  The number of the step we're at. This is used just for printing
	  informative messages
	  
	max_D:
	  Maximal bond dimension. This is used to decide when to apply the 
	  compression
	
	Output:
	-------
	glist --- The list of gates for the Trotter step.
	

	"""

	n = len(e_list)

	glist = []

	L = len(layers)

	for l in range(L):

		#
		# Apply Rx + Rz on all qubits
		#

		for i in range(n):
			glist.append(('rx', i, None, {'theta': theta_x / L}))
			glist.append(('rz', i, None, {'theta': theta_z / L}))

		#
		# Apply Rzz to the edges in the layer
		#

		for e in layers[l]:
			glist.append(('rzz', None, e, {'theta': theta_zz}))

		msg = f'\n\n            ========  Finished Step {step}  layer {l}  ========\n'
		glist.append(('*message', None, None, {'msg_str': msg}))

		#
		# Compress the TN, if needed
		#
		if 2 ** step >= D_max:
			glist.append(('*compress', None, None, None))
		else:
			msg = f'            * No need to compress (D_max={D_max}) * \n'
			glist.append(('*message', None, None, {'msg_str': msg}))

		#
		# Run a BP after the last layer is done. This is used to:
		#
		# 1) Calculate the local expectation values.
		#
		# 2) Move the system to a Vidal gauge and Calculate the edge
		#    entropy statistics.
		#
		if l == L-1:
			glist.append(('*BP', None, None, None))

			#
			# We calculate the local expectation values by calling the
			# function local_avs
			#
			params = {}
			params['ext_func'] = local_avs
			params['func_params'] = step
			glist.append(('*ext-func', None, None, params))

			params = {}
			params['stat'] = True
			glist.append(('*vgauge-noBP', None, None, params))

	return glist


#
# -----------------  create_Ising_Trotter_circ  ------------------------
#

def create_Ising_Trotter_circ(e_list, e_dict, layers, \
							  theta_x, theta_z, theta_zz, theta_init, steps, D_max=0):
	r"""

	Creates the full circuit Ising-Trotter circuit, which is made of
	steps Trotter steps, starting from a tensor product of Ry rotations.
	
	See the full circuit details in the one_Ising_Trotter_step function.
	
	Input Parameters:
	-----------------
	e_list, e_dict:
	  TN structure
	
	layers:
	  A list of layers. Each layers is the list of edges that define it.
	           
	theta_x, theta_z, theta_zz:
	  The rotations of the Trotter circuit
	  
	theta_init:
	  The angle of the initial Ry rotation. If None, then do not apply 
	  any initial rotation.
	
	steps:
	  Total number of Trotter steps
	  
	max_D:
	  Maximal bond dimension. This is used to decide when to apply the 
	  compression
	
	Output:
	-------
	glist --- The circuit's list of gates.


	"""

	glist = []

	if theta_init is not None:
		n = len(e_list)
		for i in range(n):
			glist.append(('ry', i, None, {'theta': theta_init}))

	#
	# Run a loop with step = 1,2,3,..., steps
	#
	for step in range(1, steps + 1):
		glist_trotter = one_Ising_Trotter_step(e_list, e_dict, layers, \
			theta_x, theta_z, theta_zz, step, D_max)

		glist = glist + glist_trotter

	return glist


########################################################################
#                                                                      #
#                       M A I N   P R O G R A M                        #
#                                                                      #
########################################################################


def main():

	np.seterr(all='raise')
	np.seterr(under='ignore')

	GRID = 'PRETZEL'
	
	#
	# File to hold the final PEPS TN
	#
	OUT_FNAME = 'final-PEPS.pkl'

	# 
	# "Trotter" step angles. 
	# Each step consists of 3 layers. For *each* layer we apply:
	#
	# 1. Rx(RX_ANGLE/3)   for all qubits
	# 2. Rz(RZ_ANGLE/3)   for all qubits
	# 3. Rzz(RZ_ANGLE)    for all (i,j) pairs in that layer
	#
	
	RZZ_ANGLE = 5.2
	RX_ANGLE  = 1.52
	RZ_ANGLE  = 0.87
	
	# Initial angles Ry. Put None for starting from |0>^n
	THETA_INIT = 2.6

	STEPS = 6   # Total number of steps

	D_MAX = 64  # D_MAX = chi --- the maximal bond dim
	

	L2THRESH = 1e-7  # The L_2 truncation parameter. When None, then
	                 # truncation is done only by the bond dimension.

	#
	# General BP parameters
	#
	BP_MAX_ITER = 50
	BP_DELTA    = 1e-5
	BP_DAMPING  = 0.0

	#
	# Set the numpy precision. Either double-precision 'DP'
	# or single-precision 'SP'. Note that the precision of the BP
	# iteration (which is often run on the GPU) is set separately
	# in the qbp.py file
	#

	NUMPY_PRECISION = 'SP'

	##################   Actual Program Starts Here   ####################


	TNketQC_initialize(mode=NUMPY_PRECISION)
	

	#
	# Create the IBM PRETZEL grid with 28 qubits
	#
	e_list, e_dict, angles_list, layers = create_ibm_grid(GRID)

	num_qubits = len(e_list)

	#
	# Initialize the TN to |0>^n state
	#
	T_list = create_initial_T_list(e_list, mode='|0>')

	#
	# Create the circuit we are about to simulate
	#
	glist = create_Ising_Trotter_circ(
		e_list,
		e_dict,
		layers,
		theta_x=RX_ANGLE,
		theta_z=RZ_ANGLE,
		theta_zz=RZZ_ANGLE,
		theta_init=THETA_INIT,
		steps=STEPS,
		D_max=D_MAX
	)


	print("\n\n")
	print(f"  ==========================================================")
	print()
	print(f"     Starting a PEPS-BP simulation on an IBM heavyhex grid")
	print(f"     ------------------------------------------------------")
	print()
	print(f"      Grid type: {GRID}")
	print()
	print(f"      Circuit Info:")
	print(f"      -------------")
	print()
	print(f"      (*) Number of qubits: {num_qubits}")
	print(f"      (*) Number of layers: {len(layers)}")
	print(f"      (*) Number of gates:  {len(glist)}")
	print()
	print(f"      theta_x: {RX_ANGLE:.6g}  theta_z: {RZ_ANGLE:.6g}  "\
		f"theta_zz: {RZZ_ANGLE:.6g} ")
	print(f"      Initial Ry theta: {THETA_INIT}")
	print()
	print(f"      Final PEPS TN is written to file: {OUT_FNAME}")
	print()
	print(f"  ==========================================================")
	print("\n\n")



	#
	# Initialize the TN dict
	#

	TN_params = {}
	TN_params['T_list'] = T_list
	TN_params['e_list'] = e_list
	TN_params['e_dict'] = e_dict
	TN_params['angles_list'] = angles_list
	TN_params['D_max'] = D_MAX
	TN_params['L2thresh'] = L2THRESH

	#
	# Initialize the BP dict
	#

	BP_params = {}
	BP_params['m_list']      = 'U'   # uniform initial BP messages
	BP_params['BP_max_iter'] = BP_MAX_ITER
	BP_params['BP_delta']    = BP_DELTA
	BP_params['BP_damping']  = BP_DAMPING

	#
	# Finally, apply the circuit.
	#
	T_list = apply_circuit(TN_params, BP_params, glist)

	total_err   = TN_params['total_err']
	total_f_sim = TN_params['total_f_sim']

	#
	# Save the final TN to a find
	#

	print("\n\n\n")
	print("              * * *   SIMULATION FINISHED    * * *\n\n")
	print()
	print(f"  [*] Final truncation error: {total_err:.6g}")
	print(f"  [*] Final fidelity:         {total_f_sim:.6g}")
	print()

	
	
	print(f"  Saving final TN to file {OUT_FNAME}")
	
	fout = open(OUT_FNAME, 'wb')
	pickle.dump(TN_params, fout)
	fout.close()
	
	print()
	print(f"  => Done.")

	print("\n\n")
	print("                   * * *    GOODBYE    * * * \n")


main()
