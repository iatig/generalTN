#!/usr/bin/python3

########################################################################
#                                                                      #
#               simulate-qasm-circ.py                                  #
#               ---------------------                                  #
#                                                                      #
#   Uses BP and TNketQC to simulate a quantum circuit that is          #
#   given in the file 'silly-circuit.qasm' over qubits in a PRETZEL    #
#   topology (28 qubits).                                              #
#                                                                      #
#   Once calculating the final state |psi>, calculate the local        #
#   magnetization <Z_i> for all qubits                                 #
#                                                                      #
#                                                                      #
#                                                                      #
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
	apply_circuit, calc_RDM1, circuit_from_qasm_file

from numpy import array, trace, pi

from ibm_grid import create_ibm_grid


#
# --------------------------   crossed_threshold   ---------------------
#
def crossed_threshold(es, D_dict, Dthresh, k_thresh):
	
	r"""
	
	Check if the TN needs compression. We look at the theoretical bond 
	dimension of each edge. The TN needs compression if either of the 
	following happens:
	
	1. there exists a bond > Dthresh
	2. There exist k or more bonds with bond >= Dthresh
	
	Input Paramters:
	----------------
	
	es:
	  A list of edges we check. Usually the list of edges of a particular
	  vertex.
	
	D_dict:
	  A dictionary of bond dim. D_dict[e] = bond dim of edge e
	  
	Dthresh, k_thresh:
	  Parameters defining when to compress (see explanation above)
	
	Output: True or False (whether to compress or not)
	
	"""

	bad_edges = 0
	
	for e in es:
		D = D_dict[e]
		
		if D>Dthresh:
			return True
			
		elif D==Dthresh:
			bad_edges += 1
			
	if bad_edges >= k_thresh:
		return True
		
	return False

#
# -----------------------   add_compression   --------------------------
#
def add_compression(glist, e_list, e_dict, Dmax, Dthresh, k_thresh):
	
	r"""
	
	Gets a circuit and decides when to insert a compression gate '*compress'.
	
	This is done by tracking the maximal bond dim of each edge. We define
	a bond-dim dictionary D_dict which stores this maximal bond dimension 
	for every edge e. 
	
	When a 2-local gate is applied, we update the bond dim of the edge
	it acts on. We then check if we "crossed a threshold", i.e., if 
	the bond dimension increased so much that a global compression is 
	needed. This is tested by calling the function crossed_threshold.
	
	Input Parameters:
	------------------
	glist: 
	  The circuit (list of gates)
	
	e_list, e_dict:
	  TN structure
	  
	Dmax:
	  The maxmial bond dimension of the TN *after* a compression is 
	  performed
	  
	Dthresh, k_thresh:
	  Define the criteria of when compression is need. See description 
	  in the crossed_threshold function.
	  
	Output:
	-------
	new_glist ---  An updated circuit with '*compress' gates
	
	
	"""
	
	D_dict = {}
	for e in e_dict:
		D_dict[e] = 1
		
	new_glist = []
	for l,g in enumerate(glist):
		
		gname, q, e, params = g
		
		if e is not None:
			
			if gname in ['CNOT01', 'CNOT10', 'cz', 'rzz']:
				blowup_factor = 2
			else:
				print("Unknow gate ",gname)
				exit(1)
				
			D_dict[e] = D_dict[e]*blowup_factor
			
			#
			# Check if we've crossed the threshold
			#
			
			i,i_leg, j,j_leg = e_dict[e]
			
			if crossed_threshold(e_list[i], D_dict, Dthresh, k_thresh) \
				or crossed_threshold(e_list[j], D_dict, Dthresh, k_thresh):
					
				print("Crossed A threshold:  l=",l)
				
				new_glist.append( ('*compress', None, None, None) )
				
				for e1 in e_dict:
					if D_dict[e1]>Dmax:
						D_dict[e1]=Dmax
				
			
			
			
		new_glist.append(g)
		
		
	return new_glist


#
# ----------------------  calculate_local_Z  --------------------------- 
#
def calculate_local_Z(TN_params, BP_params):
	

	T_list = TN_params['T_list']
	e_list = TN_params['e_list']
	e_dict = TN_params['e_dict']
	
	m_list = BP_params['m_list']

	Zop = array([ [1,0], [0,-1] ])
	
	Z_list = []

	for v in range(len(T_list)):
		rho1 = calc_RDM1(v, T_list, e_list, e_dict, m_list)
		
		av = trace(rho1@Zop)
		
		Z_list.append(av)
		
	return Z_list
			
	

	

# =====================================================================
#                               M A I N
# =====================================================================

def main():
	
	
	np.seterr(all='raise')
	np.seterr(under='ignore')

	QASM_FNAME  = 'silly-circuit.qasm'
	
	D_MAX = 4     # Maximal bond dim *after* compression
	D_THRESH = 8  # Maximal bond dim allowed before calling compression

	
	L2THRESH = 1e-7  # The L_2 truncation parameter. When None, then 
	                 # truncation is done only by the bond dimension.
	
	#
	# Belief Propagation parameters
	#
	BP_MAX_ITER = 50  
	BP_DELTA =1e-7
	BP_DAMPING = 0.0

	NUMPY_PRECISION = 'DP'  

	##################   Actual Program Starts Here   ####################

	TNketQC_initialize(mode=NUMPY_PRECISION)

	glist, e_list, e_dict = circuit_from_qasm_file(QASM_FNAME)
	
	#
	# Automatically add compression at various locations within the circuit
	# by looking at seeing if either:
	#
	# 1. Any of the (maximal) bond dim > D_THRESH,
	# 2. There is a vertex with 2 edges or more with bond >=D_THRESH. 
	#
	# In such case, we compress all the bonds to bond dim of at most D_MAX
	#
	
	new_glist = add_compression(glist, e_list, e_dict, Dmax=D_MAX, \
		Dthresh=D_THRESH, k_thresh=2)
	
	new_glist.append( ('*BP', None, None, None))
	
	#
	# Initialize the TN to |0>^n state
	#
	T_list = create_initial_T_list(e_list, mode='|0>')

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
	
	T_list_ket = apply_circuit(TN_params, BP_params, new_glist)
	
	TN_params['T_list'] = T_list_ket
	
	print("\n\n\n")
	print("===> Done ket evolution \n\n")
	
	
	#
	# Calculate the list of local expectation values <Z_i> and print it
	#
	Z_list = calculate_local_Z(TN_params, BP_params)
	
	for v, Z in enumerate(Z_list):
		print(f"<Z_{v}> = {Z.real:.6g}")

	
	
main()
