#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
An implementation of a thalamo-cortical conductance-based neural mass model 
using Moris Lecar equations and a neural architecture inspired by Gilbert & Weisel
and Douglas & Martin models.

Integration using a delayed Euler-like scheme and spline interpolated Fourier 
transform for spectral response. Also contains function for Dict -> vector and 
Vector -> dict conversions, default (prior / initial) parameters and an 
objective function.

See RunTCModel.py for examples of how to call the functions.

Model parameters are contained within 3 OrderedDicts:
	M = model info {sample rate & length (pst), hidden states, frequencies of interest}
	P = neuronal model parameters {S,H,T/td,CV,A,B etc.}
	G = observation model parameters
	
The Dict -> vector & Vector -> dict routines require that dicts are OrderedDicts
(import collections) rather than ordinary dicts so that the order of the keys (fields) 
are remembered - rather than returned in an arbitrary order (as per normal dicts)

AS
"""


import numpy as np
from scipy import interpolate
import collections
import scipy.io as sio
from matplotlib import pyplot as plt

__all__ = ["InitM","fx","mg_switch","Ncdf_jdw","Integrator","Observer","DictToArr"]


def InitM(M):
	# initialise states and peristimulus (sampling) time
	
	M['x'] = np.zeros(shape=[M['ns'],10,5]) # state space
	M['x'][:,:,0] = -50                     # membranes at -50
	M['pst'] = np.array(np.arange(0,M['t'],M['dt']))  # sample times
	M['fs'] = 1/M['dt']
	return M


def fx(x,u,P,M):
	# State equations for a thalamo-cortical conductance based neural mass model
	#
	# Moris-Lecar equations:
	# http://www.scholarpedia.org/article/Morris-Lecar_model
	#
	# Weisel & Gilbert / Douglas & Martin - like architecture
	# https://www.ncbi.nlm.nih.gov/pubmed/15217339
	#
	# returns x: model hidden states updated for u, P & M
	
	
	x = np.reshape(x,M['x'].shape)
	
	ns,npp,nk = x.shape
	
	# exp inputs around fixed values
	#--------------------------------------------------------------------------
	#C = np.exp(P['C'])
	G = np.exp(P['H'])
	
	# receptor time constants
	#--------------------------------------------------------------------------
	KE = np.exp(-P['T'][0][0]) * 1000/4   # AMPA - cortex
	KI = np.exp(-P['T'][0][1]) * 1000/16  # GABA-A
	KN = np.exp(-P['T'][0][2]) * 1000/100 # NMDA
	KB = np.exp(-P['T'][0][3]) * 1000/200 # GABA-B
	
	tKE = np.exp(-P['td'][0][0]) * 1000/4   # AMPA - thalamus
	tKI = np.exp(-P['td'][0][1]) * 1000/16  # GABA-A
	tKN = np.exp(-P['td'][0][2]) * 1000/100 # NMDA
	tKB = np.exp(-P['td'][0][3]) * 1000/200 # GABA-B
	
	
	
	#% Membrane capaciatances
	#--------------------------------------------------------------------------	
	CV   = np.exp(P['CV'])*[128,128,128,256,128,128,256,128,256,128]/1000
	
	GL = 1 
	
	#% Voltages
	#--------------------------------------------------------------------------
	VL   = -70;                           #  % reversal  potential leak (K)
	VE   =  60;                           #  % reversal  potential excite (Na)
	VI   = -90;                           #  % reversal  potential inhib (Cl)
	VR   = -40;                           #  % threshold potential (firing)
	VN   =  10;                           #  % reversal Ca(NMDA)   
	VB   = -100;                          #  % reversal of GABA-B
	
	#% Firing
	#--------------------------------------------------------------------------
	Vx = np.exp(P['S'])*32
	m  = Ncdf_jdw(M['x'][:,:,0],VR,Vx)
	
	#% Set up extrinsic afferents
	#--------------------------------------------------------------------------
	#% %     SP  DP  tp  rt  rc
#	SA   = [[1,   0,   0,   0,   0]   #  SS
#	        [0,   1,   0,   0,   0]   #  SP
#	        [0,   0,   0,   0,   0]   # SP2
#	        [0,   1,   0,   0,   0]   #  SI
#	        [1,   0,   0,   0,   0]   #  DP
#	        [0,   0,   0,   0,   0]   # DP2
#	        [0,   0,   0,   0,   0]   #  DI
#	        [0,   0,   0,   0,   0]   #  TP
#	        [0,   0,   0,   0,   1]   #  rt
#	        [0,   0,   0,   0,   0]]/8;#  rc
#	
#	SNMDA = SA
	
	

	"""
	intrinsic connectivity switches
	--------------------------------------------------------------------------    
	   population: 1  - Spint stellates (L4)            : 1
	               2  - Superficial pyramids (L2/3)     : 2
	               3  - Superficial pyramids (L2/3)     : 2b
	               4  - Inhibitory interneurons (L2/3)  : 3
	               5  - Deep pyramidal cells (L5)       : 4
	               6  - Deep pydamidal cells (L5)       : 4b
	               7  - Deep interneurons (L5)          : 5
	               8  - Thalamic projection neurons -L6 : 6
	               9  - Reticular cells (Thal)          : 7
	               10 - Thalamo-cortical relay cells (Thal) : 8
   """
   
   
    #% AMPA conductances
	#%      ss  sp  sp2 si  dp  dp2 di  tp  rt  rl
	GEa = np.array([[0,   0,   0,   0,   0,   0,   0,   4,   0,   4],# ss
	       [4,   0,   4,   0,   4,   0,   0,   0,   0,   0],  # sp
	       [0,   4,   0,   0,   0,   0,   0,   0,   0,   0],  # sp2
	       [4,   4,   4,   0,   0,   0,   0,   0,   0,   0],  # si
	       [0,   4,   0,   0,   0,   0,   0,   0,   0,   0],  # dp
	       [0,   4,   0,   0,   0,   0,   0,   0,   0,   0],  # dp2
	       [0,   0,   0,   0,   4,   4,   0,   0,   0,   0],  # di
	       [0,   0,   0,   0,   0,   4,   0,   0,   0,   0],  # tp
	       [0,   0,   0,   0,   0,   0,   0,   0,   0,   4],  # rt
	       [0,   0,   0,   0,   0,   0,   0,   4,   0,   0]]) # rl 
	
	#% NMDA conductances
	GEn = GEa;
	
	#% GABA-A conductances
	#%      ss,  sp  sp2 si  dp  dp2 di  tp  rt  rl
	GIa = np.array([[8,   0,   0,   8,   0,   0,   0,   0,   0,   0],  # ss
	       [0,   8,   0,   8,   0,   0,   0,   0,   0,   0],  # sp
	       [0,   0,   8,   8,   0,   0,   0,   0,   0,   0],  # sp2
	       [0,   0,   0,   32,  0,   0,   0,   0,   0,   0],  # si
	       [0,   0,   0,   0,   8,   0,   8,   0,   0,   0],  # dp
	       [0,   0,   0,   0,   0,   8,   8,   0,   0,   0],  # dp2
	       [0,   0,   0,   0,   0,   0,   32,  0,   0,   0],  # di
	       [0,   0,   0,   0,   0,   0,   8,   8,   0,   0],  # tp
	       [0,   0,   0,   0,   0,   0,   0,   0,   32,  0],  # rt
	       [0,   0,   0,   0,   0,   0,   0,   0,   8,   32]])#% rl
	
	#% GABA-B conductances
	GIb = GIa;
   

	#% Background activity and exogenous inputs
	#--------------------------------------------------------------------------
	BE = np.exp(P['E'])*0.8	

	U  = u
	
	f  = x.copy()
	
	# Flow / dynamics
	
	#for i in range(npp):
	i = 0
	
	# intrinsic coupling
	E = np.inner(G[:,:,i] * GEa, m)
	EN = np.inner(G[:,:,i] * GEn, m)
	I = np.inner(G[:,:,i] * GIa, m)
	IB = np.inner(G[:,:,i] * GIb, m)
	
	# background activity
	E = E + BE
	EN = EN + BE
	
	# exogenous input
	E[9] = E[9] + U
	EN[9] = EN[9] + U
	
	# voltage
	f[:,:,i] = GL*(VL - x[i,:,0]) + \
				x[i,:,1] * (VE - x[i,:,0]) + \
				x[i,:,2] * (VI - x[i,:,0]) + \
				x[i,:,4] * (VB - x[i,:,0]) + \
				x[i,:,3] * (VN - x[i,:,0]) * mg_switch(x[i,:,0]) / CV
	
	# conductance	- cortical		 
	f[i,0:7,1] = (np.transpose(E[0:7]) - x[i,0:7,1])*KE
	f[i,0:7,2] = (np.transpose(I[0:7]) - x[i,0:7,2])*KI
	f[i,0:7,4] = (np.transpose(IB[0:7]) - x[i,0:7,4])*KB
	f[i,0:7,3] = (np.transpose(EN[0:7]) - x[i,0:7,3])*KN
	 
	# conductance   - thalamic
	f[i,8:9,1] = (np.transpose(E[8:9]) - x[i,8:9,1])*tKE
	f[i,8:9,2] = (np.transpose(I[8:9]) - x[i,8:9,2])*tKI
	f[i,8:9,4] = (np.transpose(IB[8:9]) - x[i,8:9,4])*tKB
	f[i,8:9,3] = (np.transpose(EN[8:9]) - x[i,8:9,3])*tKN

	return f


def mg_switch(V):
	#% switching output s: determined by voltage (V) depedent magnesium blockade
	#% parameters as per Durstewitz, Seamans & Sejnowski 2000
	
	s = 1.50265/(1 + 0.33*np.exp(-0.06*V));
	return s

def Ncdf_jdw(x,u,v):
	# Cumulative distribution function for univariate normal distributions 
	# - J.D. Williams approximation
	x = (x - u) / np.sqrt(np.abs(v))
	F = np.sqrt(1 - np.exp(-(2/np.pi)*np.square(x)))/2
	i = x < 0
	F[i] = -F[i]
	F = F + 0.5
	return F

def Integrator(P,M,U,G):
	# A Euler-like numerical integration scheme with simple delays
	#
	#
	#
	#
	#
	
	pst = M['pst']
	
	# input
	Mu = M['pst']*0
	Mu = Mu + np.exp(G['a'])
	
	# delays
	#D  = float(1)
	D = np.array([1,16])
	d  = -D*np.exp(P['D'])/1000
	npp = 10
	Sp = np.kron(np.ones([5,5]),np.kron( np.eye(npp,npp),np.eye(1,1))) #% states: same pop.
	Ss = np.kron(np.ones([5,5]),np.kron( np.ones([npp,npp]),np.eye(1,1))) #% states: same source
	
	Dp = np.logical_not(Ss)                      # states: different sources
	Ds = np.logical_and(np.logical_not(Sp), Ss)  # states: same source different pop.
	
	thal = [0,0,0,0,0,0,0,0,1,1] # treat thalamic states as extrinsic
	thal = np.outer(thal,thal)
	thal = np.kron(np.ones([5,5]),np.kron(thal,np.eye(1,1)))
	Dp   = np.logical_or(Dp, thal)
	D    = d[1]*Dp + d[0]*Ds
	
	dt = M['dt']
	
	x = M['x']
	v = x
	y = np.zeros(shape=[50,len(pst)])
	Firing = np.zeros(shape=[10,len(pst)])

	# integrate
	for i in range(len(M['pst'])):
		dxdt = fx(v,Mu[i],P,M)
		
		delay = np.reshape( np.inner(D*dt,dxdt.flatten()), dxdt.shape)
		v = v + dt*(dxdt + delay)
		y[:,i] = v.flatten()
	
		VR = -40
		Vx = np.exp(P['S'])*32
		m  = Ncdf_jdw(M['x'][:,:,0],VR,Vx)
		Firing[:,i] = m
	
	return y, Firing
	
	
def Observer(x,G):
	# Simple observation model (weighting of states)
	#
	#
	#
	#
	#
	#
	
	J = G['J']
	y = np.inner(np.transpose(J),np.transpose(x))
	y = y * G['L']
	return y


def SpectralResponse(X0,M,w):
	# Spectral response from a neural mass using Fourier transform and cubic spline interpolation
	#
	#
	#
	#
	#
	#
	
	X1 = np.fft.fft(X0)
	X1 = X1.real
	
	n = X1.size
	dt = M['dt']
	freq = np.fft.rfftfreq(n, d=dt)
	
	q = X1.shape[1] / 2
	
	if (freq.size == q + 1):
		q = q + 1
	
	X1 = X0[:,0:q]
	X1 = np.abs(X1.transpose())
	
	
	# spline
	S = interpolate.splrep(freq, X1)
	xnew = interpolate.splev(w,S)
	
	return xnew


def RunIntRespond(M,P,G):
	# A wrapper on the above to initiate, integrate, observe (weight) and 
	# compute spectral response from a neural mass model
	#
	#
	#
	#
	
	
	# Complete model coniguration
	M = InitM(M)
	
	# call integration scheme
	y0,f0 = Integrator(P,M,1,G)
	
	# weighted signal
	X0 = Observer(y0,G)
	
	# frequencies to return
	#w = np.linspace(4, 100, num=97, endpoint=True) # freqs of interest
	w = M['w']
	
	YY = SpectralResponse(X0,M,w)

	return YY, w


def DictToArr(M):
	# Convert a dict containing multiple numpy arrays to one continuous numpy array
	# Can also contain sub-dicts.
	# ALL DICTS MUST BE OrderedDicts  !!
	V = np.empty(0)
	
	for i in M.keys():
		if type(M[i]) == np.ndarray:
			V = np.append(V,M[i])
			#mask = np.ones(len(V), dtype=bool)
			#mask[0] = False
			#V = V[mask]
			
		elif type(M[i]) == collections.OrderedDict:
			V0 = DictToArr(M[i])
			V = np.append(V,V0)
	return V
			

def ArrToDict(V,M):
	# Given a template dict M, un-vectorise numpy array V in its fields 
	# - even if it holds sub-dicts
	# ALL DICTS MUST BE OrderedDicts  !!
	N = collections.OrderedDict()
	for i in M.keys():
		X = M[i]
		if type(X) == np.ndarray:
			S = X.size
			N[i] = np.reshape(V[0:S],X.shape)
		
			# remove first N points of array
			mask = np.ones(len(V), dtype=bool)
			mask[:S] = False
			V = V[mask]
		elif type(X) == collections.OrderedDict:
			N[i],V = ArrToDict(V,M[i])
	return N, V

def LoadDCM(filein):
	M = sio.loadmat(filein)
	D = M['D']
	y = D['y'] # the spectral density of a sensor
	w = D['w'] # the corresponding frequencies
	y = y[0][0]
	w = w[0][0]
	return y,w

def ObjectiveFun(V,PG,M,File):
	# Objective function for the complete model
	#
	# Pass V - vector version of model dicts P & G
	#     PG - dict version of model param dicts P & G
	#      M - model structure M
	#      File = fullfile to .mat with empirical data in
	
	P0,z = ArrToDict(V,PG)
	
	
	P = P0['P']
	G = P0['G']
	
	
	y,f = Integrator(P,M,1,G) # integrate
	X0 = Observer(y,G) # weight
	w = M['w']
	y = SpectralResponse(X0,M,w) # compute spectral repsonse
	
	y0,w0 = LoadDCM(File)
	
	e = np.square(np.sum(y0 - y))
	
	#plt.plot(w0,y0,w,y)
	
	return e
	
def DefaultParams():
	# Initial flat (0) parameter priors and other defaults inc.
	#
	# 1s sampling time
	# 4 - 80 Hz fitting
	#
	#
	M =	collections.OrderedDict()
	M["ns"]   = np.array(1)
	M["dt"]   = np.array(0.0033)
	M["t"]    = np.array(1)
	M["w"]   = np.linspace(4, 80, num=77, endpoint=True) # freqs of interest
	
	
	# Neural mass parameters  - P
	#------------------------------------------------------------------------------
	P =	collections.OrderedDict()
	P["S"]   = np.zeros(shape=[M['ns'],10])      # firing per pop
	P["H"]   = np.zeros(shape=[10,10,M['ns']])   # intrinsic connections
	P["T"]   = np.zeros(shape=[M['ns'],4])       # cortical receptor TCs
	P["td"]  = np.zeros(shape=[M['ns'],4])       # thalamus receptor TCs
	P["C"]   = np.zeros(shape=[M['ns'],1])       # exogenous inputs
	P["A"]   = np.zeros(shape=[M['ns'],M['ns']]) # extrinsic connectivity
	P["B"]   = np.zeros(shape=[M['ns'],M['ns']]) # trial specific extrinsics
	P["CV"]  = np.zeros(shape=[M['ns'],10])      # membrane capacitance 
	P["D"]   = np.array([0,0])                   # delays [intr, extr]
	P["E"]   = np.array(0)                       # background activity
	
	
	# Observation parameters  - G
	#------------------------------------------------------------------------------
	G = collections.OrderedDict()
	G["L"]   = np.array(0)
	G["J"]   = np.zeros(shape=[M['ns']*10*5,1])
	G["a"]   = np.array(0)
	
	
	
	# contributing cells and weights
	G['J'][0:10,0] = [.2,.8,.8,0,.2,.2,0.,2,0,0]
	
	return M,P,G
	
	
def NewParams():
	# Slightly more informed starting points for parameters
	#
	# 1s sampling time
	# 4 - 80 Hz fitting
	#
	#
	
	M,P,G = DefaultParams()
	

	# Neural mass parameters  - P
	#------------------------------------------------------------------------------
	P['S'][:,0:10] = [0.2126,1.3472,0.3523,3.5484,0.4020,0.3627,0.3577,0.3014,0.2972,0.2525]
	P["H"]   = np.zeros(shape=[10,10,M['ns']])   # intrinsic connections
	
	P["T"][:,0:4]  = [-0.5221,0.2270,-0.0456,-0.4284]
	P["td"][:,0:4] = [-0.1038,-0.1267,-0.0110,0.0656]
	
	P["A"]   = np.zeros(shape=[M['ns'],M['ns']]) # extrinsic connectivity
	P["B"]   = np.zeros(shape=[M['ns'],M['ns']]) # trial specific extrinsics
	P["CV"][:,0:10] = [-0.4041,-0.4461,-0.0939,-0.1505,-0.1518,0.1114,0.1133,0.0922,-0.0997,0.0271]
	P["D"] = np.array([2.0437e-05,1.7290e-07])
	P["E"] = np.array(-0.0404)                       # background activity
	
	
	# Observation parameters  - G
	#------------------------------------------------------------------------------
	G["L"]   = np.array(-6.3197)
	G["J"]   = np.zeros(shape=[M['ns']*10*5,1])
	G["a"]   = np.array(0.3701)
	
	
	
	# contributing cells and weights
	G['J'][0:10,0] = [-0.4757,0.8000,0.8000,0,-0.5813,-0.5687,0,-0.2548,0,0]	
	
	
	P["H"][0:10,0:10,0] = [[1.1587,    0     ,    0     ,    0.1639,    0     ,   0     ,  0     ,  0.0477,  0     , -0.0664],
				          [0.6371,   -0.0438,   -0.0228,   -0.0622,    0.0199,   0     ,  0     ,  0     ,  0     ,  0],
					      [0     ,    0.1919,    0.0564,   -0.0900,    0     ,   0     ,  0     ,  0     ,  0     ,  0],
					      [0.1046,    0.3744,    0.1269,   -0.3070,    0     ,   0     ,  0     ,  0     ,  0     ,  0],
		                  [0     ,    0.0505,    0     ,    0     ,   -0.2998,   0     , -0.0156,  0     ,  0     ,  0],
				          [0     ,   -0.2491,    0     ,    0     ,    0     ,   0.1664,  0.0248,  0     ,  0     ,  0],
				          [0     ,    0     ,    0     ,    0     ,    0.1830,   0.0959,  0.2613,  0     ,  0     ,  0],
				          [0     ,    0     ,    0     ,    0     ,    0     ,  -0.1527,  0.2147,  0.8646,  0     ,  0],
				          [0     ,    0     ,    0     ,    0     ,    0     ,   0     ,  0     ,  0     , -0.3497,  0.0444],
				          [0     ,    0     ,    0     ,    0     ,    0     ,   0     ,  0     , -0.1174,  0.1099,  0.2597]]
	return M,P,G
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	
	