#!/usr/bin/python3

########################################################################
#
#
#                   square_grid.py
#                   =================
#
# Contains basic functions to manipulate the TN params of a square
# grid, either we open or periodic boundary conditions.
#
#
# An example of an open BC 3x4 grid with its edges is:
#
#   (0)-- e0-R --(1)-- e1-R --(2)-- e2-R --(3)
#    |            |            |            |
#   e0-D         e1-D         e2-D         e3-D 
#    |            |            |            |
#   (4)-- e4-R --(5)-- e5-R --(6)-- e6-R --(7)
#    |            |            |            |
#   e4-D         e5-D         e6-D         e7-D 
#    |            |            |            |
#   (8)-- e8-R --(9)-- e9-R --(10)--e10-R--(11)
#
#
#
#
#
# History:
# ---------
#
# 5-Aug-2025  Itai  Initial version.
#
# 14-Aug-2025 Itai  Added optional edge parameter to get_v_edges()
#
# 17-Nov-2025 Itai  Add periodic-BC support
#
#
########################################################################


from qbp import calc_e_dict
from numpy import pi


#
# ---------------------- get_v_idx  ------------------------
#

def get_v_idx(i,j,Nx):

	"""

	Get the index of a tensor on the square grid using its
	(i,j) location (row i, column j)

	For example, a 3x4 grid is arranged as:

  (0) ---- (1) ---- (2) ---- (3)
	 |        |        |        |
  (4) ---- (5) ---- (6) ---- (7)
	 |        |        |        |
  (0) ---- (1) ---- (2) ---- (3)

	Input Parameter
	---------------

	i,j --- The (i,j) location
	Nx --- Linear horizontal size of the grid (no. of columns)

	Output: the index v := i*Nx + j
	--------

	"""

	ij = i*Nx + j

	return ij


#
# ---------------------- get_row_and_col  ------------------------
#

def get_row_and_col(v, Nx):
	"""
	
	Given the index v of a site, finds its corresponding row and column.
	
	Input Parameters:
	------------------
	v --- The index number
	Nx --- The x dimension of the grid
	
	Output:
	-------
	i,j --- The col,row of v
	
	"""
	i = v // Nx
	j = v - i*Nx

	return i,j




#
# -------------------------  get_v_edges  ----------------------------
#

def get_v_edges(v, Nx, Ny, edge=None, BC='OO'):
	
	"""
	
	Given a vertex index v, find the label of one or all of its edges, 
	which correspond to e_list[v].
	
	The edges are sorted by (R, L, U, D)
	
	If the vertex is on the boundary then one or more of these edges
	will be absent (open boundary conditions)
	
	The labels of the edges are always defined to be:
	(*) e<v>-R for the horizontal edge pointing to right of v
	(*) e<v>-D for the vertical edge pointing downward of v
	
	For example, for a 3x4 square grid 
	
	0---1---2---3
	|   |   |   |
	4---5---6---7
	|   |   |   |
	8---9---10--11
	
	The edges of vertex 5 are: ('e4-R', 'e5-R', 'e1-D', 'e5-D')

	Input Parameters:
	------------------
	v      --- The index of the vertex
	
	Nx, Ny --- Dimensions of the grid
	
	edge   --- (optional) the name of the edge ('L','R','U','D'). If 
	           None is given, then output a list of all edges.
	
	BC     --- Boundary condition in X and Y dimensions. It is a 2-letters
	           string. 'PO' means periodic in X, Open in Y. 'PP' means
	           periodic in both dims, etc..
	
	Output Parameters:
	------------------
	
	es --- The list of edges labels of the vertex
	
	
	
	"""
	
	i,j = get_row_and_col(v,Nx)
	
	es = []
	
	if j>0:
		ij = get_v_idx(i,j-1,Nx)
		eL = f'e{ij}-R'
		
		if edge=='L':
			return eL
			
		es.append(eL)
		
	elif BC[0]=='P':
		# so j=0 and we're on periodic x BC
		ij = get_v_idx(i,Nx-1,Nx)
		eL = f'e{ij}-R'
		
		if edge=='L':
			return eL
			
		es.append(eL)
		
	
	if j<Nx-1 or BC[0]=='P':
		eR = f'e{v}-R'

		if edge=='R':
			return eR
			
		es.append(eR)

		
	if i>0:
		ij = get_v_idx(i-1,j,Nx)
		eU = f'e{ij}-D'

		if edge=='U':
			return eU
			
		es.append(eU)
		
	elif BC[1]=='P':
		# so i=0 and we're on periodic y BC
		ij = get_v_idx(Ny-1,j,Nx)
		eU = f'e{ij}-D'

		if edge=='U':
			return eU
			
		es.append(eU)

		
	if i<Ny-1 or BC[1]=='P':
		eD = f'e{v}-D'
		
		if edge=='D':
			return eD
			
		es.append(eD)
	
	if edge is not None:
		print(f"Error in get_v_edges(): non-existent edge {edge} for v={v}")
		exit(1)
	
	return es
		
	
#
# -------------------------  get_v_angles  ----------------------------
#

def get_v_angles(v, Nx, Ny, BC='OO'):

	"""
	
	Given a vertex index v, find the angles of its edges, which 
	correspond to angles_list[v].
	
	The edges are sorted by (R, L, U, D) and therefore the corresponding
	angles are 0, pi, pi/2, 3pi/2.
	
	If the vertex is on the boundary then one or more of these angles
	will be absent.


	Input Parameters:
	------------------
	v      --- The index of the vertex
	Nx, Ny --- Dimensions of the grid
	
	BC     --- Boundary condition in X and Y dimensions. It is a 2-letters
	           string. 'PO' means periodic in X, Open in Y. 'PP' means
	           periodic in both dims, etc..

	Output Parameters:
	------------------
	
	angles --- The list of angles of the vertex
	

	
	"""

	
	i,j = get_row_and_col(v,Nx)
	
	angles = []
	
	if j>0 or BC[0]=='P':
		angles.append(pi)
	
	if j<Nx-1 or BC[0]=='P':
		angles.append(0)
		
	if i>0 or BC[1]=='P':
		angles.append(pi/2)
		
	if i<Ny-1 or BC[1]=='P':
		angles.append(1.5*pi)

	return angles
		
	
	


#
# -------------------------   create_square_grid   ---------------------
#

def create_square_grid(Nx, Ny, BC='OO'):

	r"""

	Construct the e_list, e_dict of a square grid of size Nx\times Ny
	with either open or periodic boundary conditions

	Input Parameters:
	-----------------
	Nx, Ny --- dimensions of the grid
	
	BC     --- Boundary condition in X and Y dimensions. It is a 2-letters
	           string. 'PO' means periodic in X, Open in Y. 'PP' means
	           periodic in both dims, etc..

	Output:
	-------

	e_list, e_dict --- The data structure of the TN graph

	"""

	e_list = []
	angles_list = []

	for i in range(Ny):
		for j in range(Nx):
			
			v = get_v_idx(i,j,Nx)
			es =get_v_edges(v, Nx, Ny, BC=BC)
			angles = get_v_angles(v, Nx, Ny, BC=BC)
			
			e_list.append(es)
			angles_list.append(angles)

	e_dict = calc_e_dict(e_list)

	return e_list, e_dict, angles_list


			
#
# -------------------    print_peps_shape    --------------------------
# 
def print_peps_shape(T_list, Nx, Ny=None):

	r"""
	
	A very basic printing of the bond dimensions of the tensors in the
	square grid
	
	Input Parameters:
	------------------
	T_list --- the list of tensors in the square grid TN
	Nx, Ny --- the dimensions of the grid
	
	"""

	if Ny is None:
		Ny = len(T_list)//Nx
	
	print()
	print(f"            square PEPS shape  ({Ny} x {Nx})")
	print("          -------------------------------------\n")
		
	
	for i in range(Ny):
		s = ''
		
		for j in range(Nx):
			ij = get_v_idx(i,j,Nx)
			sh = list(T_list[ij].shape)
			sh = sh[1:]
			s = s + f' {sh}'
		print(s)
		
	print()
	
