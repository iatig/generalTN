#!/usr/bin/python3

########################################################################
#
#
#                        1d-TFIM-evolution.py
#                        ====================
#
# Use PEPS-BP to simulate a 1d circuit that of a Trotter-Suzuki evolution
# of a transverse Ising model
#
# History:
# --------
#
# 24-Mar-2026:  Initial version
#
#
########################################################################


from os import environ, uname
N_THREADS = '4'
#environ['OMP_NUM_THREADS'] = N_THREADS

environ["OMP_NUM_THREADS"] = N_THREADS # export OMP_NUM_THREADS=4
environ["OPENBLAS_NUM_THREADS"] = N_THREADS # export OPENBLAS_NUM_THREADS=4 
environ["MKL_NUM_THREADS"] = N_THREADS # export MKL_NUM_THREADS=6
environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS # export VECLIB_MAXIMUM_THREADS=4
environ["NUMEXPR_NUM_THREADS"] = N_THREADS # export NUMEXPR_NUM_THREADS=6


import sys
import time
import pickle

import numpy as np

from qbp import calc_e_dict

from TNketQC import TNketQC_initialize, create_initial_T_list, \
	apply_circuit, calc_RDM1

from numpy import array, trace, pi



#
# -----------------------  create_1D_grid  --------------------------
#
def create_1D_grid(n):
	r"""
	
	Create the TN structure parameters for a 1D lattice of n qubits.
	
	The edges are 'e0-1', 'e1-2', ...
	
	They are partitioned into two layers:
	Layer-0: ['e0-1', 'e2-3', 'e4-5', ...]
	Layer-1: ['e1-2', 'e3-4', 'e5-6', ...]
	
	The edges of each vertex i are ordered as: [e{i-1}-{i}, e{i}-{i+1}]
	
	
	
	"""
	
	e_list = []
	
	for i in range(n):
		es = []
		if i>0:
			es = es + [f'e{i-1}-{i}']
		if i<n-1:
			es = es + [f'e{i}-{i+1}']
			
		e_list.append(es)
		

	#
	# Now partition the edges into even/odd layers
	#
	layer_even = []
	layer_odd = []
	
	for i in range(n-1):
		e = f'e{i}-{i+1}'
		
		if i % 2 ==0:
			layer_even.append(e)
		else:
			layer_odd.append(e)
			
	layers = [layer_even, layer_odd]
	
	e_dict = calc_e_dict(e_list)
	
	return e_list, e_dict, layers





#
# ---------------------- calc_average_magnetization --------------------
#

def calc_average_magnetization(TN_params, BP_params, glist, t,\
	func_params):
	
	r"""
	
	This function is called from within the circuit to calculate the
	average magnetization. 
		
	M := \frac{1}{N} \sum_i <Z_i>
	
	at various steps of the circuit.
	
	This function has the general structure that is required by 
	TNketQC.apply_circuit().

	
	Input Parameters:
	------------------
	TN_params, BP_params --- TN & BP params
	
	glist       --- The circuit we're simulating
	
	t           --- The gate number we're in
	
	func_params --- general parameters that are passed to the function.
	                In this case, func_params holds the Trotter-Suzuki
	                step from which the function was called.
	
	
	Output:
	-------
	T_list,err --- Updated T_list and error (not used here).
	
	"""
	
	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~   calc_Z
	#
	def calc_Z(i):
		r"""
		
		A small internal function to calculate the <Z> expectation value
		at site i
		
		"""
		
		sigma_Z = array([[1,0],[0,-1]])
		
		rho_i = calc_RDM1(i, T_list, e_list, e_dict, m_list)
		
		avZ = trace(rho_i@sigma_Z)
		
		return avZ.real



	#
	# ~~~~~~~~~~~~~~~~~~~~~~~  function starts here  ~~~~~~~~~~~~~~~~~~~~~
	#

	T_list = TN_params['T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']
	m_list = BP_params['m_list']
	
	step = func_params # which step we're in

	#
	# Calculate the average total magnetization:
	#
	#  \frac{1}{n} \sum_{i=1}^n <Z>_i
	#
	
	n = len(T_list)
	M = 0
	for i in range(n):
		Z = calc_Z(i)
		M +=  Z
		
	M = M/n
	
	print()
	print(f"Averages: M({step}) = {M}")
	print()
	
	#
	# Return the same T_list and 0 error (because we did not change the
	# TN)
	#
	
	return T_list, 0


#
# ----------------------  one_Ising_Trotter_step  ----------------------
#
def one_TFIM_step(e_list, e_dict, layers, j_zz, h_x, step, D_max=0):
	
	"""
	Applies a single TFIM step:
	
	1. Apply e^{-i h_x X} to all qubits
	2. Go over the layers, and in each layer apply e^{-i j_zz ZZ}
	
	A compression step is added (only) when the bond dimension possibly
	passes D_max (i.e, when 2^steps > D_max).
	
	Input Parameters:
	-----------------
	e_list, e_dict --- TN structure
	
	layers         --- list of layers. Each layer is the list of edges in it
	
	j_zz, h_x      --- The parameters of the TFIM circuit
	
	step           --- Which step we are in 
	
	D_max          --- Max bond dimension. When 2^steps > D_max, we apply 
	                   compression.
	
	"""
	
	n = len(e_list)
	
	glist = []
	
	
	num_layers = len(layers)
	
	#
	# First, apply e^{-i 2 h_x X} to all qubits
	#
	for i in range(n):
		glist.append( ('rx', i, None, {'theta': 2*h_x}) )
	
	#
	# Next, go over all layers, and in each layer apply e^{-i 2 h_zz ZZ}
	# to all edges
	#
	for l in range(num_layers):
						
		for e in layers[l]:
			glist.append( ('rzz', None, e, {'theta': 2*j_zz}) )

		msg = f'\n\n            ========  Finished layer {l} in Step {step}  ========\n'
		glist.append( ('*message', None, None, {'msg_str':msg}) )

		#
		# Compress the TN, if needed
		#
		if 2**step>=D_max:
			glist.append(('*compress', None, None, None))
		else:
			msg = f'            * No need to compress (D_max={D_max}) * \n'
			glist.append( ('*message', None, None, {'msg_str':msg}) )
		
		#
		# Run a BP after the 3rd layer is done. This is used to:
		#
		# 1) Calculate the local expectation values.
		#
		# 2) Move the system to a Vidal gauge and Calculate the edge 
		#    entropy statistics.
		#
		if l==num_layers-1:
			glist.append(('*BP', None, None, None))
			
			#
			# We calculate the local expectation values by calling the 
			# function local_avs
			#
			params={}
			params['ext_func'] = calc_average_magnetization   
			params['func_params'] = step
			glist.append(('*ext-func', None, None, params))
			
			params = {}
			params['stat']=True
			glist.append(('*vgauge-noBP', None, None, params))
			
	return glist
			

#
# -----------------  create_Ising_Trotter_circ  ------------------------
#

def create_TFIM_circ(e_list, e_dict, layers, j_zz, h_x, steps, D_max=0):
		
	"""
	
	Creates the full TFIM circuit circuit, which is made of steps Trotter
	steps.
	
	"""
	
	
	glist = []
	
	# Run a loop with step = 1,2,3,..., steps
	for step in range(1, steps+1):
		glist_trotter = one_TFIM_step(e_list, e_dict, layers, \
			j_zz, h_x, step, D_max)
		
		glist = glist + glist_trotter
		
		
	return glist
	




########################################################################
#                                                                      #
#                       M A I N   P R O G R A M                        #
#                                                                      #
########################################################################


def main():
	
	global qubits_map, qbits_sset
	

	np.seterr(all='raise')
	np.seterr(under='ignore')


	# TFIM parameters
	
	N = 50              # Number of qubits in the 1D system
	
	#
	# Two qubits gate applied is e^{-i JZZ_ANGLE Z\otimes Z}
	#
	JZZ_ANGLE = pi/4     

	#
	# 1 qubit gate applied is e^{-i JX_ANGLE X}
	#
	HX_ANGLE = 0.45*pi
	
	                 
	STEPS  = 10      # Total number of Trotter-Suzuki steps
	
	D_MAX  = 32     # D_MAX = chi --- the maximal bond dim
	
	L2THRESH = 1e-7  # The L_2 truncation parameter. When None, then 
	                 # truncation is done only by the bond dimension.
	
	BP_MAX_ITER = N+10  # In a 1D system BP always converges after at most
	                    # N steps
	BP_DELTA =1e-7
	BP_DAMPING = 0.0
	
	#
	# Set the numpy precision. Either double-precision 'DP'
	# or single-precision 'SP'. Note that the precision of the BP
	# iteration (which is often run on the GPU) is set separately 
	# in the qbp.py file
	#
	
	NUMPY_PRECISION = 'DP'  

	##################   Actual Program Starts Here   ####################

	TNketQC_initialize(mode=NUMPY_PRECISION)

	#
	# Create the 1D grid with N qubits
	#
	e_list, e_dict, layers = create_1D_grid(N)
	

	# ====================================================================
	# -------------------- forward state propagation ---------------------
	# ====================================================================

	
	#
	# Initialize the TN to |0>^n state
	#
	T_list = create_initial_T_list(e_list, mode='|0>')

	#
	# Create the circuit we are about to simulate
	#
	glist = create_TFIM_circ(e_list, e_dict, layers, \
		JZZ_ANGLE, HX_ANGLE, STEPS, D_max=D_MAX)
	
	#
	# Initialize the TN dict
	#

	TN_params={}
	TN_params['T_list'] = T_list
	TN_params['e_list'] = e_list
	TN_params['e_dict'] = e_dict
	TN_params['D_max']  = D_MAX
	TN_params['L2thresh'] = L2THRESH

	#
	# Initialize the BP dict
	#

	BP_params = {}
	BP_params['m_list'] = 'U'
	BP_params['BP_max_iter'] = BP_MAX_ITER
	BP_params['BP_delta'] = BP_DELTA
	BP_params['BP_damping'] = BP_DAMPING
	
	
	#
	# Finally, apply the circuit.
	#
	T_list_ket = apply_circuit(TN_params, BP_params, glist)
	
	print("\n\n\n")
	print("===> Done ket evolution \n\n")
	
	
main()
