# -*- coding:Utf-8 -*-
#####################################################################
#This file is part of RGPA.

#Foobar is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#Foobar is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

#-------------------------------------------------------------------
#authors :
#Nicolas AMIOT		: niamiot@univ-rennes1.fr
#Bernard UGUEN		: buguen@univ-rennes1.fr
#Mohamed LAARAIEDH	: mohamed.laaraiedh@univ-rennes1.fr
#####################################################################
import numpy as np
import scipy as sp 
from copy import copy
from BoxN import *
from RSSLocation import *
#from Model import *
from Constraint import *


class RSS(Constraint):	
	"""	RSS Constraint

	Description and evaluation of TOA constraints

	:Parameters:
		value	: float
			Constraint value in dB. Default = 40
		std	: float
			Value standard deviation in dB. default = 1.0
		vcw	: float
			scale factor. Default = 1.0
		model	: dictionnary
			keys:	self.model['PL0'] =-34.7
				self.model['d0']  = 1.0 
				self.model['RSSnp'] = 2.64	
				self.model['RSSStd'] = 4.34 
				self.model['Rest'] = 'mode'
		p	: np.array 1 x ndim
			constraint center
		
	:Attributes:

		range	: distance conversion from time self.value.
		sstd	: distance conversion from time self.std
		runable	: True NOT USED
		evaluated :False NOT USED		
		self.Id	: Constraint ID

		from annulus bound:
		min	: minimum value of observable
		max	: maximum value of observable
		mean	: mean value of observable


	:Methods:
		annulus_bound(self)	: Compute the minimum and maximum distance of the enclosing annulus of the constraint 
		rescale(self,vcw)	: rescale contraint boundary with a given scale factor 'vcw'
		inclusive(self,b)	: Is constraint center is inside a box ?
		valid(self,b)		: Test if Lbox is compatible with the constraint
		valid_v(self,lv)	: Test if a liste of a vertexes from a box is compatible with the constraint. vertexes are obtained thanks to LBoxN.bd2coordinates()
		estvol(self)		: Constraint Volume estimation 

	"""

	def __init__(self,value=40,std=1.0,vcw=3,model={},p=np.array([])):	
		Constraint.__init__(self,'RSS',p)

		self.Id     = copy.copy(self.C_Id)
		Constraint.C_Id = Constraint.C_Id+1   # constraint counter is incremented
		self.value  = value  # attennation (dB)
		if len(model)==0:
			self.model['PL0'] =-34.7
			self.model['d0']  = 1.0 
			self.model['RSSnp'] = 2.64
			self.model['RSSStd'] = 4.34 
			self.model['Rest'] = 'mode'
		else	:
			self.model  = model

		self.LOC = RSSLocation(p)
		self.std  = (self.LOC.getRangeStd(p, self.model['PL0'], self.model['d0'], self.value,self.model['RSSnp'], self.model['RSSStd'], self.model['Rest'])[0])/0.3
		self.range    = self.LOC.getRange(p, self.model['PL0'], self.model['d0'], self.value, self.model['RSSnp'], self.model['RSSStd'], self.model['Rest'])[0]
		self.sstd   = self.std*0.3
		self.rescale(vcw)
		self.runable = True
		self.evaluated = False
		self.annulus_bound()
		
	def annulus_bound(self):
		"""
		annulus_bound():
		Compute the minimum and maximum distance of the enclosing annulus of the constraint

		:Returns:
			Nothing but update cmin, cmax 
		"""
		self.cmin = max(0,self.range - self.vcw*self.sstd	)
		self.cmax = self.range + self.vcw*self.sstd	

  	def rescale(self,vcw):
		"""rescale bounds of the constraint with vcw factor
	
		:Parameters:
			vcw : float
				scale factor
		
		"""
		self.vcw = vcw
		dx          = self.range+(self.vcw*self.std)*0.3*np.ones(len(self.p))
		box         = BoxN(np.array([self.p-dx,self.p+dx]),ndim=len(self.p))

		self.lbox   = LBoxN([box])
		self.estvol()

	def inclusive(self,b):
		""" Is constraint center is inside a box ?
		:Parameters:
			b	: LBoxN

		:Returns:
			Boolean

		"""
		if b.inbox(self.p):
			return True
		else:
			return False

	

	def valid(self,b):
		"""check if a box is valid for the given constraint 

		:Parameters:
			b	: LBoxN

		:Return:
			True	: if box is enclosed
			False	: if the box is ambiguous
			'Out'	: if the box is out
		
		"""
		p0 = self.p
		v  = self.bd2coord(b)
		P  = np.outer(np.ones(len(v)),p0)
		D  = P-v
		DD = np.sqrt(np.sum(D*D,axis=1))

	        DDcmin = sum(DD>=cmin)
		DDcmax = sum(DD<=cmax)
	
		if DDcmin+DDcmax > 15:
			return(True)
		elif (DDcmin <1) | (DDcmax <1) :#(bmax<cmin)|(bmin>cmax): 

			return('out')			
		else :
			return(False)


	def valid_v(self,v):
		"""
		.. todo:
			update as TOA valid_v !!
		
		valid_v(v) : check if vertex are valid for the given constraint 
		
		A box is valid if it not not valid

		A box is not valid if all distances are greater than rangemax
			           or all distances are less than rangemin				
		"""
		p0 = self.p
		P  = np.outer(np.ones(len(v)),p0)
		D  = P-v
		DD = np.sqrt(np.sum(D*D,axis=1))
		DD2 = (DD>=self.cmin) & (DD<=self.cmax)
	        
		return DD2
	def estvol(self):
		""" Constraint Volume estimation 

		"""
		R2 = self.range+self.vcw*self.sstd
		R1 = self.range-self.vcw*self.sstd
		V2 = (4.*np.pi*R2**3)/3
		V1 = (4.*np.pi*R1**3)/3
		self.estvlm = V2-V1

