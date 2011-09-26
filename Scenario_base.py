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
import time 
import pdb
import os
import sys

import matplotlib.pyplot as plt


sys.path.append('./Util') 

from BoxN import *
import Geomview as g
from Scene import *

sys.path.append('./Algebraic_loc') 
from RSSLocation  import *
from ToALocation  import *
from TDoALocation import *
from HDFLocation  import *
from PyGraphTool import cdf


sys.path.append('./Constraints') 
from EXCLUDE import *
from RSS import *
from TOA import *
from TDOA import *
from CLA import *



class Scenario(object):
        """
        Class Scenario : definition of a static localization scenario

        MEMBERS

        an      : Anchor nodes coordinates
        bn      : Blind nodes coordinates
        CDF     : list 
        parmsc  : parameters dictionary 
                scenar_type     :  'simulated' | 'monte-carlo'
                err_type        :  'homogene'  | 'hybrid'
                Constrain_Type  :  'TOA' | 'TDOA' | 'RSS' | 'MIX'
                Algebraic_method     :  'TLS' | 'WLS'
                rss_mode        :  'mean'
                'rss0'          :  -34.7 dB
                'd0'            :  1 m 
                'pn'            :  2.64
                'std_dev_range' :  arange(1,5,.5)
                'dec_mode'      : 'monodec'
                'vcw'           : validity constraint width : 3
                'sigma_max'     : 3
                'an_nb'         : Number of used anchor nodes
                'std_dev_arange' :  arange(1,5,.5)
                'eval_pos'      : True
                'algebraic'     : True

##      parmsc_dis  : display parameters dictionary 
##
##      METHODS
##      
##      info        :  Display information about the scenario 
##      show3(ibn)  :  3D Geomview display ibn : index blind node 

    
        """
        def __init__(self,an,bn,parmsc={},parmsh={}):

                self.an    = an 
                self.bn    = bn
                self.std_v = np.zeros(len(an))
                self.CDF   = []
                self.std_dev_arr_compt=0
                self.vcw_arr_compt=0
		self.ndim  = len(self.an[0].T)
                if len(parmsc)==0:
                        # parameter of scenario
                        self.parmsc={}
                        self.parmsc['scenar_type']='simulated'#'monte_carlo'  # choose the type of scenario : Simulated : give some AN and BN. 
                        self.parmsc['err_type']='homogene' # homogene/hybrid = error applied on AN
                        parmsc['exclude_out']= np.array((np.min(self.an,axis=0),np.max(self.an,axis=0))) # look for estimated position inside the area created by the AN 
                        self.parmsc['Constrain_Type']='TOA'  # choose the type of constrain to compute between BN:TOA/TDOA/RSS/MIX
                        ### TDOA parmsc
                        self.parmsc['Algebraic_method']='TLS'  # if ['Constrain_Type']='TDOA' chose the TDOA method WLS/TLS
                        ## RSS parmsc
                        self.parmsc['rss_mode']='mean'
                        self.parmsc['rss0']=-34.7
                        self.parmsc['d0']=1.0
                        self.parmsc['pn']=2.64
                        self.parmsc['dec_mode']='monodec'  # choose the type of constrain to compute between BN:TOA/TDOA/RSS/MIX
                        self.parmsc['vcw'] = 3.0 # validity constrain width
                        self.parmsc['sigma_max']=7.0 # choose random standard deviation to apply to measures
                        self.parmsc['sigma_max_RSS'] =5.0
                        self.parmsc['sigma_max_TOA'] =3.0
                        self.parmsc['sigma_max_TDOA'] =3.0
                        self.parmsc['an_nb']=len(self.an) # choose the number of AN for performing positionning
                        self.parmsc['std_dev_arange']=np.arange(1.,5.,.5) # standart deviation range for monte carlo computing 
                        # monte carlo 3D parameters
                        self.parmsc['vcw_arange']=np.arange(1.,4.,0.5) # validity constraint width (fourchette) 
                        self.parmsc['eval_pos']=True # if True , the position of the last evaluated layer is estimated
                        self.parmsc['algebraic']=True # compute algebraic method 
                else:
                        self.parmsc = parmsc



                if len(parmsh) == 0:
                        self.parmsh['display']=False     # launch geomview interactively 
                        self.parmsh['scene']=True      # display whole scene
                        self.parmsh['boxes']=True       # display constraint box
                        self.parmsh['constr_boxes']=False       # display constraint box
                        self.parmsh['estimated']=True  # display estimated point
                else : 
                        self.parmsh=parmsh
                # display parameters
                self.parmsc_dis={}
                self.parmsc_dis['CDF_bound']=np.arange(0,5.0,0.01)
                self.parmsc_dis['CDF_bound_auto_adapt']=True
                self.parmsc_dis['room']=[]

                if self.parmsc['scenar_type']=='simulated':

                        self.parmsc_dis['display']=True
                        self.parmsc_dis['save_fig']=False

                elif self.parmsc['scenar_type']=='monte_carlo':

                        self.parmsc_dis['display']=False
                        self.parmsc_dis['save_fig']=True

        
                else :

                        self.parmsc_dis['display']=True
                        self.parmsc_dis['save_fig']=False



                if self.parmsc_dis['display']==False : 
                        plt.rcParams['xtick.labelsize']='x-large'
                        plt.rcParams['ytick.labelsize']='x-large'
                        plt.rcParams['axes.labelsize']= 'large'
                        plt.rcParams['font.weight']='normal'
                        plt.rcParams['xtick.minor.size']=2
                        plt.rcParams['legend.fontsize']= 'xx-large' 
                        plt.rcParams['font.size']=20
                        plt.rcParams['grid.linewidth']=3.5
                        plt.rcParams['xtick.major.pad']=20
                

                if self.parmsc_dis['save_fig']:
                        self.fig = plt.figure(figsize=(28, 20))
                        self.ax = self.fig.add_subplot(111)
                else:
                        self.fig = plt.figure()         
                        self.ax = self.fig.add_subplot(111)
                self.marker=[[':r',':g',':b',':k',':m',':c'],['--r','--g','--b','--k','--m','--c'],['-r','-g','-b','-k','--m','--c']]
                
                if self.parmsc['Constrain_Type']=='TDOA':
                        self.dan=[]
                        for i in range(len(self.an)-1):
                                self.dan.append(vstack((an[0],an[i+1])))

                

        def info(self):
                """

                info : display information about current scenario

                """
                print "-------------------------------"
                print "     SCENARIO INFORMATION"
                print "-------------------------------"
                print "scenario type   : ",self.parmsc['scenar_type']
                print "constraints type  : ",self.parmsc['Constrain_Type']


                #### NOEUDS
                if self.parmsc['Constrain_Type']=='TDOA':
                        print "nb of AN couple : ", len(self.dan)
                else :
                        print "nb AN  : ", len(self.an)
                
                print "nb BN  : ", len(self.bn)
                print 'limit of BNs positions',self.parmsc_dis['room'],'m'


                if self.parmsc['scenar_type']=='simulated':

                        ##### geometric computaton
                        print 'decimation method : ',self.parmsc['dec_mode']
                        if self.parmsc['Constrain_Type']=='TOA':
                                print 'validity constraint width :',self.parmsc['vcw'],'m'

                        print "Error type   : ",self.parmsc['err_type']
                        print "std dev : ",self.parmsc['sigma_max'],'ns'
                        print "std dev : ",self.parmsc['sigma_max_RSS'],'ns'
                        print "std dev : ",self.parmsc['sigma_max_TOA'],'ns'
                        print "std dev : ",self.parmsc['sigma_max_TDOA'],'ns'
                        print "std vect : ",self.std_v,'ns'

                        ########## algebraic
                        print "algebraic compute : ",self.parmsc['algebraic']
                        if self.parmsc['algebraic']:
                                if self.parmsc['Constrain_Type']=='TDOA':
                                        print 'algebraic TDOA method :',self.parmsc['Algebraic_method']  # if ['Constrain_Type']='TDOA' chose the TDOA method WLS/TLS
                                        
                        ####### RSS
                        if self.parmsc['Constrain_Type']=='RSS':
                                print 'rss mode :',self.parmsc['rss_mode']
                                print 'RSS0 :',self.parmsc['rss0']
                                print 'd0 : ',self.parmsc['d0']
                                print 'np : ',self.parmsc['pn']


                #############" MONTE CARLO


                if self.parmsc['scenar_type']=='monte_carlo':
                        lcdf=len(self.CDF)
                        print '\n'
                        for i in range(lcdf):
                                print 'computation nb',i+1
                                if self.CDF[i]['mode']=='algebraic':
                                        print 'algebraic'
                                        print '\n'
                                else :
                                        print 'err_type :',self.CDF[i]['err_type']
                                        print 'vcw :',self.CDF[i]['vcw'],'m'
                                        print 'decimation :',self.CDF[i]['dec']
                                        print 'sigma max :',self.CDF[i]['sigma_max'],self.std_v,'ns'
                                        print '\n'
                                        


                
        def show3(self,amb=True,sc='all'):
                """
                show3(param) : Geomview 3D vizualization of the scenario

                The scene is stored in the file scene.list in the geom directory 
                
                param :
                        display : True
                        R       : Sphere radius (meter) 

                """
		self.cla.show3(amb=amb,sc=sc)


        def run(self):
                """

                run : run simulation of the scenario  

                """

                self.CRB=[]
                Nbn    = len(self.bn)
                Nan    = len(self.an)
                if self.parmsc['save_pe']:
                        self.pe   = []
                        self.p_LS = []
                        self.p_LSci = []
                        self.p_LScr = []
                if self.parmsc['save_CLA']:
                        self.CLA = []

                self.err1    = np.array([])
                #self.err2    = np.array([])
                self.errx    = np.array([])
                self.erry    = np.array([])
                self.errz    = np.array([])
                self.errLS   = np.array([])
                self.errLSci   = np.array([])
                self.errLScr   = np.array([])
                self.errxLS  = np.array([])
                self.erryLS  = np.array([])
                self.errzLS  = np.array([])
                # list for algebraic 
        
                atoabn = []
                astdbn = []
                P      = []
                
                if self.parmsc['Constrain_Type']=='TDOA':
                        lbcl=Nan-1
                else :                          
                        lbcl=Nan

                tdoa_idx = nonzero(np.array(l_connect)=='TDOA')[0]

                if self.parmsc['Constrain_Type']=='hybrid' or self.parmsc['Constrain_Type']=='test':
                        algehyb=1
                else :
                        algehyb=0


                if len(tdoa_idx) != 0:
                        lbcl=Nan-1

                self.rss_idx = nonzero(np.array(l_connect)=='RSS')[0]
                self.toa_idx = nonzero(np.array(l_connect)=='TOA')[0]
                self.tdoa_idx = nonzero(np.array(l_connect)=='TDOA')[0]
                self.ERRSAVE=[]
		self.trgpa=[]
		self.talg=[]
                for ibn in range(Nbn):              # for each blind node

			

                        self.errRSS=zeros((1))
                        self.errTOA=zeros((1))
                        self.errTDOA=zeros((1))
                        self.ibn=ibn
                        errli   = []
                        Constraint.C_Id = 0             # reset constraint Id for each BN
                	
                        print "                                    Blind Node N°",ibn+1,"/",Nbn
                        atv=[]
                        pbn = self.bn[ibn] 
                        #print "Blind Node N° ",ibn,pbn
                        
                        self.tic_ensemblist=time.time()
                        cla = CLA(self.parmsh)
                        cla.bn = self.bn[ibn] 

                        clarss = CLA(self.parmsh)
                        clatoa = CLA(self.parmsh)
                        clatdoa = CLA(self.parmsh)

                

                        if parmsc['exclude_out'] != None :
                                E = Exclude(nodes=parmsc['exclude_out'])        
                                cla.append(E)   


                        for ian in range(lbcl):        # for each AN or couple of AN (TDOA)
                        	TC=0
                                try :
                                        self.parmsc['Constrain_Type'] = self.parmsc['l_connect'][ian]
                                except :
                                        pass
                                

	####################### TOA : Geometric description 
                                if self.parmsc['Constrain_Type']=='TOA':

                                        pan    = self.an[ian] 
                                        # tvalue : true value (ns)
                                        tvalue = np.sqrt(np.dot(pan-pbn,pan-pbn))/3e8
                                        # err (ns)

                                        err    = (self.std_v[ibn,ian]*sp.randn())
                                        while err + tvalue < 0:
                                                err    = (self.std_v[ibn,ian]*sp.randn())
					self.ERRSAVE.append(err)
                                        self.errTOA=np.vstack((self.errTOA,err))
                                        # value (toa : ns )


                                        value  = max(0,tvalue+err)
					TC1=time.time()
                                        C      = TOA(value=value*1e9,
                                                     std=self.std_v[ibn,ian]*1e9,
                                                     vcw=self.parmsc['vcw'],
                                                      p=pan)

                                        cla.append(C)
					TC2=time.time()
                                        clatoa.append(C)


	####################### TDOA : Geometric description 
                                if self.parmsc['Constrain_Type']=='TDOA':


                                        
                                        pan = vstack((self.an[tdoa_idx[0]],self.an[ian+1]))
                                        # dan : delay between 2 AN (ns)
                                        dan = np.sqrt(dot(pan[0]-pan[1],pan[0]-pan[1]))/3e8

                                        minvalue = -dan
                                        maxvalue =  dan
                                        toa1 = np.sqrt(np.dot(pan[0]-pbn,pan[0]-pbn))/3e8
                                        toa2 = np.sqrt(np.dot(pan[1]-pbn,pan[1]-pbn))/3e8
                                        tvalue = toa2-toa1

                                        err    = self.std_v[ibn,ian]*sp.randn()
                                        while ((tvalue + err) < minvalue) or ((tvalue + err) > maxvalue):
                                                err    = self.std_v[ibn,ian]*sp.randn()
      					self.ERRSAVE.append(err)

                                        self.errTDOA=np.vstack((self.errTDOA,err))
                                        # tvalue : true value (ns)

#                                       value  = max(minvalue,tvalue + err)
#                                       value  = min(maxvalue,value)

                                        # value (ns)
                                        value = tvalue+err
					TC1=time.time()

                                        C    = TDOA(value=-value*1e9,
                                                    std=(self.std_v[ibn,ian]*1e9),
                                                    vcw=self.parmsc['vcw'],
                                                    p = pan)
                                        cla.append(C)
					TC2=time.time()
                                        #C    = TDOA(p=pan,value=value,std=2)
                                        clatdoa.append(C)


	####################### RSS : Geometric description 
                                if self.parmsc['Constrain_Type']=='RSS':

                                        pan = self.an[ian] 
                                        d0     =  self.parmsc['d0']
                                        RSSnp  =  vstack((self.parmsc['pn'],self.parmsc['pn']))
                                        PL0= vstack((self.parmsc['rss0'],self.parmsc['rss0']))
                                        RSSStd=  vstack((self.parmsc['sigma_max_RSS'],self.parmsc['sigma_max_RSS']))

                                        PP= vstack((self.bn[ibn],self.bn[ibn]))
                                        PA= vstack((pan,pan))
                                        rssloc = RSSLocation(PP)
                                        value = (rssloc.getPLmean(PA.T, PP.T, PL0, d0, RSSnp))
                                        err = (RSSStd*randn(shape(value)[0],shape(value)[1]))[0][0]
					self.ERRSAVE.append(err)
                                        self.errRSS=np.vstack((self.errRSS,err))
                                
                                        value = value[0] + err
                        
#                                       value  = rssloc.getPL(PA.T, PP.T, PL0, d0, RSSnp, RSSStd)
                                        value  = value


#                                       value = self.parmsc['rss0']-10*self.parmsc['pn']*log10(dr/self.parmsc['d0'])+self.err_v[ibn,ian]

                                        self.Model = {}
                                        self.Model['PL0'] =self.parmsc['rss0']
                                        self.Model['d0']  = self.parmsc['d0'] 
                                        self.Model['RSSnp'] = self.parmsc['pn']
                                        self.Model['RSSStd'] = self.parmsc['sigma_max_RSS']
                                        self.Model['Rest'] = 'mode'
					TC1=time.time()
                                        C   = RSS(value=value,
                                                  std=self.std_v[ibn,ian],
                                                  vcw=self.parmsc['vcw'],
                                                  model=self.Model,
                                                  p=pan )
                                        cla.append(C)
					TC2=time.time()
                                        clarss.append(C)
				TC=TC+(TC2-TC1)
                                if self.parmsc['algebraic']:
                                                atv.append(value)

                        if len(self.rss_idx) != 0:
                                self.errRSS = delete(self.errRSS,0,0)
                        if len(self.toa_idx) != 0:
                                self.errTOA = delete(self.errTOA,0,0)
                        if len(self.tdoa_idx) != 0:
                                self.errTDOA = delete(self.errTDOA,0,0)
                        # version boite recursive       
                        # 
                        ######################### CLA TOTALE
			TR1=time.time()
			cla.merge2()
                        cla.refine(cla.Nc)
                        self.cla = cla
                        ### DoubleListRefine version
                        cla.estpos2()
                        pe1=cla.pe
			TR2=time.time()
			self.trgpa.append((TR2-TR1)+TC)
                        self.pe.append(pe1)
                        errli.append(np.sqrt(np.dot(pe1[:2]-pbn[:2],pe1[:2]-pbn[:2])))
                        err1=min(errli)

                        self.err1  = np.hstack((self.err1,err1))        

                 
                        if self.parmsc['algebraic']:

                                if algehyb==1:
                                        self.parmsc['Constrain_Type']='hybrid'


                                p_LS    = self.algebraic_compute(atv,None)
#                                p_LSci    = self.algebraic_compute(atv,self.pe[-1])
#                                p_LScr    = self.algebraic_compute(atv,self.bn[ibn])

                                if self.parmsc['save_pe']:
                                        self.p_LS.append(p_LS)
#                                        self.p_LSci.append(p_LSci)
#                                        self.p_LScr.append(p_LScr)
                                if self.parmsc['Constrain_Type'] != 'hybrid':
                                        errLS   = np.sqrt(np.dot(p_LS[:2]-pbn[:2],p_LS[:2]-pbn[:2]))
#                                        errLSci   = np.sqrt(np.dot(p_LSci[:2]-pbn[:2],p_LSci[:2]-pbn[:2]))
#                                        errLScr   = np.sqrt(np.dot(p_LScr[:2]-pbn[:2],p_LScr[:2]-pbn[:2]))                

     
        
                                else :
                                        
                                        errLS   = np.sqrt(np.dot(p_LS[:2]-pbn[:2],p_LS[:2]-pbn[:2]))
#                                        errLSci   = np.sqrt(np.dot(p_LSci[:2]-pbn[:2],p_LSci[:2]-pbn[:2]))                        
#                                        errLScr   = np.sqrt(np.dot(p_LScr[:2]-pbn[:2],p_LScr[:2]-pbn[:2]))                        
                                self.errLS  = np.hstack((self.errLS,errLS))     
#                                self.errLSci  = np.hstack((self.errLSci,errLSci))
#                                self.errLScr  = np.hstack((self.errLSci,errLScr))          

				if errli > errLS:
					print 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNnn\n',errli,'\n',errLS

			

        def algebraic_compute(self,atv,P0init):
                
                rss_idx = self.rss_idx
                toa_idx = self.toa_idx
                tdoa_idx = self.tdoa_idx

                P=[]

#               
#               print 'RSS_idx',rss_idx
#               print 'TOA_idx',toa_idx
#               print 'TDOA_idx',tdoa_idx


                ############### RSS ##############
                RN_RSS=zeros(self.ndim)
                Rss=zeros(1)
                RSSStd=zeros(1)

                d0     = self.parmsc['d0']
                RSSnp  = self.parmsc['pn']*ones((len(self.an[rss_idx]),1))      
                PL0=self.parmsc['rss0']*ones((len(self.an[rss_idx]),1))
                RSSStd= self.parmsc['sigma_max_RSS']*ones((len(self.an[rss_idx]),1))



                if len(rss_idx) != 0:
                
                        
                        for i in rss_idx:

                                aa=np.array(self.an[i])
                                ss=np.array(atv[i])


                                RN_RSS=np.vstack((RN_RSS,aa))                           
                                Rss=np.vstack((Rss,ss))                         


                        RN_RSS=np.delete(RN_RSS,0,0).T
                        RN_RSS = RN_RSS[:2]                             
                        Rss=np.delete(Rss,0,0)



                        

                else :

                        RN_RSS=None

                
                
                ############### TOA ##############
                RN_TOA=zeros(self.ndim)
                ToA=zeros(1)
                ToAStd=zeros(1)

                if len(toa_idx) != 0:


                        for i in toa_idx:

                                aa=np.array(self.an[i])
                                ss=np.array(atv[i])
                                tt=np.array(self.std_v[self.ibn,i])

                                RN_TOA=np.vstack((RN_TOA,aa))                           
                                ToA=np.vstack((ToA,ss))                         
                                ToAStd=np.vstack((ToAStd,tt))

                        RN_TOA=np.delete(RN_TOA,0,0).T
                        RN_TOA = RN_TOA[:2]
                        ToA=np.delete(ToA,0,0)
                        ToAStd=np.delete(ToAStd,0,0)
                        


                else : 
                        RN_TOA=None


                ############### TDOA ##############
                RN_TDOA=zeros(self.ndim)
                RN_TDOA_ref=zeros(1)
                TDoA=zeros(1)
                TDoAStd=zeros(1)

                if len(tdoa_idx) != 0:
                        #RN_TDOA=zeros(3)
                        
                        #for i in tdoa_idx:

                        #       aa=np.array(self.an[i])
                        #       RN_TDOA=np.vstack((RN_TDOA,aa)) 
                        RN_TDOA=(self.an[tdoa_idx[1:]]).T
                        RN_TDOA_ref=(self.an[tdoa_idx[0]]*ones(np.shape(RN_TDOA.T))).T            
                        


                        for i in tdoa_idx[0:-1]:

                                ss=np.array(atv[i])
                                tt=np.array(self.std_v[self.ibn,i])


                                TDoA=np.vstack((TDoA,ss))
                                TDoAStd=np.vstack((TDoAStd,tt))
                        TDoA=np.delete(TDoA,0,0)
                        TDoAStd=np.delete(TDoAStd,0,0)
                        RN_TDOA = RN_TDOA[:2]
                        RN_TDOA_ref = RN_TDOA_ref[:2]

                else :

                        RN_TDOA=None
                        RN_TDOA_ref=None
                
                      
                self.tic_algebric=time.time()
                S1=HDFLocation(RN_RSS, RN_TOA, RN_TDOA)
                S2=RSSLocation(RN_RSS)
                S3=ToALocation(RN_TOA)
                S4=TDoALocation(RN_TDOA)
		TA1=time.time()
                if self.parmsc['Algebraic_method'] == 'LS':
                        print 'to be implemented'

                elif self.parmsc['Algebraic_method'] == 'TLS':
                        print 'to be implemented'

                elif self.parmsc['Algebraic_method'] == 'WLS':
                        print 'to be implemented'

                elif self.parmsc['Algebraic_method'] == 'TWLS':
                        if RN_RSS ==None and RN_TOA==None:
                                P=S4.TWLSTDoALocate(RN_TDOA,RN_TDOA_ref, TDoA, TDoAStd)
                        elif RN_RSS==None and RN_TDOA==None:
                                P=S3.TWLSToALocate(RN_TOA,ToA, ToAStd)
                        elif RN_TOA==None and RN_TDOA==None:
				print RN_RSS, PL0, d0, Rss, RSSnp, RSSStd,
                                P=S2.TWLSRSSLocate(RN_RSS, PL0, d0, Rss, RSSnp, RSSStd, 'mode')
                        else:
                                P=S1.TWLSHDFLocate(RN_RSS, RN_TOA, RN_TDOA,RN_TDOA_ref, ToA, ToAStd, TDoA, TDoAStd, PL0, d0, Rss, RSSnp, RSSStd, 'mode')

                elif self.parmsc['Algebraic_method'] == 'ML':
			if P0init == None:
		                PP=L*rand(2,1)  # for 3D replace by L*rand(3,1)                    
		                P0=L*rand(2,1)  # for 3D replace by L*rand(3,1)                    
			else : 
				P0=P0init.reshape(2,1)
				PP=P0init.reshape(2,1)
#                        P0[2]=0.0     # for 3D uncomment
                        if RN_RSS ==None and RN_TOA==None:
                                P=S4.MLTDoALocate(PP, P0, RN_TDOA,RN_TDOA_ref, TDoA, TDoAStd)
                        elif RN_RSS==None and RN_TDOA==None:
                                P=S3.MLToALocate(PP, P0, RN_TOA,ToA, ToAStd)
                        elif RN_TOA==None and RN_TDOA==None:
                                P=S2.MLDRSSLocate(PP, P0, RN_RSS, PL0, d0, Rss, RSSnp, RSSStd)
                        else:
                                P=S1.MLHDFLocate(PP, P0, RN_RSS, RN_TOA, RN_TDOA,RN_TDOA_ref, ToA, ToAStd, TDoA, TDoAStd, PL0, d0, Rss, RSSnp, RSSStd, 'mode')

                elif self.parmsc['Algebraic_method'] == 'SDP':
                        if RN_RSS ==None and RN_TOA==None:
                                P=S4.SDPTDoALocate(RN_TDOA,RN_TDOA_ref, TDoA, TDoAStd)
                        elif RN_RSS==None and RN_TDOA==None:
                                P=S3.SDPToALocate(RN_TOA,ToA, ToAStd)
                        elif RN_TOA==None and RN_TDOA==None:
                                P=S2.SDPRSSLocate(RN_RSS, PL0, d0, -Rss, RSSnp, RSSStd, 'mode')
                        else:
                                P=S1.SDPHDFLocate(RN_RSS, RN_TOA, RN_TDOA,RN_TDOA_ref, ToA, ToAStd, TDoA, TDoAStd, PL0, d0, Rss, RSSnp, RSSStd, 'mode')
               

                      


                else : 
                        print "You have to chose the self.parmsc['Algebraic_method'] between those choices :\n LS,TLS,WLS,TWLS,ML,SDP "


                if self.parmsc['CRB'] :

                        CRBL=CRBLocation(None)
                                                        
                        if len(rss_idx) != 0:   
                                RSSStdX = self.errRSS#[rss_idx]#*ones(len(self.an[rss_idx]))

                        if len(toa_idx) != 0:   
                                TOAStdX =self.errTOA#[toa_idx]#*ones((len(self.an[toa_idx]),1))

                        if len(tdoa_idx) != 0:  
                                TDOAStdX =self.errTDOA#[tdoa_idx]#*ones((len(self.an[tdoa_idx])-1,1))

                        PP=self.bn[self.ibn,:2]


###################################### RSS PUR
                        if RN_TOA==None and RN_TDOA==None: 
                                print 'RSS'
                                self.CRB.append(sqrt(CRBL.CRB_RSS_fim(PP, RN_RSS, RSSnp,RSSStdX)))
###################################### TOA PUR
                        elif RN_RSS==None and RN_TDOA==None:  # TOA
                                print 'TOA CRB'
                                
                                self.CRB.append(sqrt(CRBL.CRB_TOA_fim(PP, RN_TOA,TOAStdX)))

###################################### TDOA PUR
                        elif RN_RSS ==None and RN_TOA==None : # TDOA
                                print 'TDOA'
                                self.CRB.append(sqrt(CRBL.CRB_TDOA_fim(PP,  RN_TDOA,RN_TDOA_ref,TDOAStdX)))
                       
                        elif RN_TOA==None and RN_TDOA!= None:
###################################### TDOA 
                                if RN_RSS==None: 
                                        print 'TDOA2'
                                        self.CRB.append(sqrt(CRBL.CRB_TDOA_fim(PP, RN_RSS, PL0, RSSStdX)) )
###################################### RSS + TDOA
                                else : 
                                        print 'RSS+TDOA'
                                        self.CRB.append(sqrt(CRBL.CRB_RSS_TDOA_fim(PP, RN_RSS, RN_TDOA,RN_TDOA_ref,RSSnp, RSSStdX, TDOAStdX ))) 

                        
                        elif RN_TOA!=None and RN_TDOA== None:
##################################### TOA
                                if RN_RSS==None: 
                                        print 'TOA'
                                        self.CRB.append(sqrt(CRBL.CRB_TOA_fim(PP, RN_TOA,TOAStdX)))
##################################### RSS + TOA
                                else :
                                        print 'RSS + TOA'
                                        self.CRB.append(sqrt(CRBL.CRB_RSS_TOA_fim(PP, RN_RSS,RN_TOA, RSSnp,  RSSStdX, TOAStdX)))

                        elif RN_TOA!=None and RN_TDOA!= None:
##################################### TOA+TDOA
                                if RN_RSS==None :
                                        print 'TOA + TDOA'
                                        self.CRB.append(sqrt(CRBL.CRB_TOA_TDOA_fim(PP, RN_TOA, RN_TDOA, RN_TDOA_ref,TOAStdX,TDOAStdX)))

##################################### RSS+TOA+TDOA
                                else :
                                        print 'RSS + TOA + TDOA'
                                        self.CRB.append(sqrt(CRBL.CRB_RSS_TOA_TDOA_fim(PP, RN_RSS, RN_TOA, RN_TDOA, RN_TDOA_ref, RSSnp, RSSStdX, TOAStdX, TDOAStdX)))


                P=array(P)
		TA2=time.time()
		self.talg.append(TA2-TA1)
		return P[:,0]




        
        def CDF_figure_gen(self,in_cdf,c_nb):
                
                
                bound=self.parmsc_dis['CDF_bound']


                if in_cdf['mode']=='algebraic':
                        cdf=in_cdf['cdf_alg']
                else:
                        cdf=in_cdf['cdf']
                
        
                

                 #size ko               
                room=self.parmsc_dis['room']
                lbound=len(bound)
                cmpt=in_cdf['cmpt']
                
                c=('c' +str(c_nb))

                if self.parmsc['scenar_type']=='simulated' :
                        
                        c=self.ax.plot(bound,cdf[:lbound],self.marker[cmpt][c_nb],linewidth=3)
                return c

                

        def CDFdisplay(self):
                title = 'CDF : ' + self.parmsc['Constrain_Type'] + '\n'+'for '+str(len(self.bn)) +' random BNs positions '   +'\n' +r'  $\sigma_{max}(RSS)$=' +str(self.parmsc['sigma_max_RSS']) +'m $\sigma_{max}(TOA)$=' +str(self.parmsc['sigma_max_TOA']) +' m ' +'$\sigma_{max}(TDOA)$=' +str(self.parmsc['sigma_max_TDOA']) +' m ' 


                leg_alg   = 'Algebraic methode :' +self.parmsc['Algebraic_method'] 
                leg2   = 'Geometric Box' +r'  $\sigma_{max} RSS$=' +str(self.parmsc['sigma_max_RSS']) +'TOA' +str(self.parmsc['sigma_max_TOA']) +'TDOA' +str(self.parmsc['sigma_max_TDOA']) +' ns ' +str(self.std_v) +'  '  +'  vcw:' +str(self.parmsc['vcw'])
                
                LEG = 'Hybrid ensemblist' 
                
                ld = []
                """
                d0 = {}
                d0['values'] = self.err2
                d0['bound']  = np.linspace(0,max(self.err2),100)
                d0['xlabel'] = 'distance (m)' 
                d0['ylabel'] = 'Cummulative density function' 
                d0['legend'] = 'Method Grid'
                d0['title']  = title  
                d0['marker'] = 'r-'  
                d0['linewidth'] = 3  
                d0['filename']  = 'essai.png'  
                ld.append(d0)
                """
                d2 = {}
                d2['values'] = self.err1
                d2['bound']  = np.linspace(0,20,100)#np.linspace(0,max(self.err1),100)
                d2['xlabel'] = 'distance (m)' 
                d2['ylabel'] = 'Cummulative density function' 
                d2['legend'] = LEG#'Method Box'
                d2['title']  = title  
                d2['marker'] = 'g-'  
                d2['linewidth'] = 3  
                d2['filename']  = 'essai.png'  
                ld.append(d2)
                
                if self.parmsc['algebraic']:
                        d1 = {}
                        d1['values'] = self.errLS
                        d1['bound']  = np.linspace(0,20,100)#np.linspace(0,max(self.errLS),100)
                        d1['xlabel'] = 'distance (m)' 
                        d1['ylabel'] = 'Cummulative density function' 
                        d1['legend'] = leg_alg
                        d1['title']  = 'title'  
                        d1['marker'] = 'b-'  
                        d1['linewidth'] = 3  
                        d1['filename']  = 'essai.png'  
                        ld.append(d1)
#               c1 = CDF.CDF(ld)
#               c1.show()
		        


                cdf(self.err1,"r-","RGPA",2)
		if self.parmsc['algebraic']:
		#np.save('rgpa1ns',self.err1)
		        cdf(self.errLS,"g-",self.parmsc['Algebraic_method'],2)
		  #      cdf(self.errLSci,"b-","ML CI",2)
              #  cdf(self.errLScr,"k-.",r"$\sqrt{CRLB}$",2)
#                cdf(self.CRB,"g-.",r"$\sqrt{CRLB}$",2)
                plt.legend(loc=4,numpoints=1)
                plt.axis([0,10,0,1])
                plt.grid('on')
                plt.xlabel("Positioning error (m)")
                plt.ylabel("Cumulative probability")
                plt.savefig(filename, format="pdf")
                plt.close()

#                file.write("PA   "+str(median(self.err1))+"\n")
#                file1.write("Geo   "+str(mean(self.err1))+"\n")
#                file1.write("ML   "+str(mean(self.errLS))+"\n")
#                file1.write("CRB   "+str(mean(self.CRB))+"\n")

        def compute(self):
                """

                compute : start the simulation of the current scenario 

                """
                self.std_v_save=[]

                        
                if self.parmsc['scenar_type']=='simulated':
                        if self.parmsc['err_type']=='homogene':
                                if self.parmsc['Constrain_Type'] != 'hybrid':

                                        self.std_v=self.parmsc['sigma_max']*sp.ones(len(self.bn),len(self.an))
                                        self.std_v_save.append(self.std_v)
                                
                                else :


                                        self.std_v=zeros((len(self.bn),len(self.an)))
                                        pstdv=nonzero(array(self.parmsc['l_connect'])=='RSS')[0]
                                        self.std_v[:,pstdv]=self.parmsc['sigma_max_RSS']*sp.ones(len(self.bn),len(self.an[pstdv]))
                                        pstdv=nonzero(array(self.parmsc['l_connect'])=='TOA')[0]
                                        self.std_v[:,pstdv]=self.parmsc['sigma_max_TOA']*sp.ones(len(self.bn),len(self.an[pstdv]))
                                        pstdv=nonzero(array(self.parmsc['l_connect'])=='TDOA')[0]
                                        self.std_v[:,pstdv]=self.parmsc['sigma_max_TDOA']*sp.ones(len(self.bn),len(self.an[pstdv]))


                        elif self.parmsc['err_type']=='hybrid':
                                if self.parmsc['Constrain_Type'] != 'hybrid':
					random.seed(seed=5)
                                        self.std_v=self.parmsc['sigma_max']*sp.rand(len(self.bn),len(self.an))-self.parmsc['sigma_max']/2.
                                        self.std_v_save.append(self.std_v)

                                else :


                                        self.std_v=zeros((len(self.bn),len(self.an)))
                                        pstdv=nonzero(array(self.parmsc['l_connect'])=='RSS')[0]

                                        self.std_v[:,pstdv]=(self.parmsc['sigma_max_RSS'])*sp.ones((len(self.bn),len(self.an[pstdv])))
                                        pstdv=nonzero(array(self.parmsc['l_connect'])=='TOA')[0]

                                        self.std_v[:,pstdv]=(self.parmsc['sigma_max_TOA']/3e8)*sp.ones((len(self.bn),len(self.an[pstdv])))
                                        pstdv=nonzero(array(self.parmsc['l_connect'])=='TDOA')[0]
                                        self.std_v[:,pstdv]=(self.parmsc['sigma_max_TDOA']/3e8)*sp.ones((len(self.bn),len(self.an[pstdv])))

                        

                        else :
                                print 'compute() : non-valid err_type'

                        cdfbound = max([self.parmsc['sigma_max_RSS'],self.parmsc['sigma_max_TOA'],self.parmsc['sigma_max_TDOA']])

#                       self.parmsc_dis['CDF_bound']=np.arange(0,cdfbound+3.,0.01)
                        self.parmsc_dis['CDF_bound']=np.arange(0,20,0.01)
                        self.run()
                        self.CDFdisplay()

                

                 
if __name__=="__main__":
		
#	try:
#		import psyco
#		psyco.full()
#  	except ImportError:
#        	pass


	os.system('killall gvx')
        L  = 20
        H1 = 0.0
        H2 = H1
        H3 = H1
        
        Nbn  = 1000
#        file=open('Time_Geo.txt', 'w')
#        file1=open('Geo_vs_ML.txt', 'w')
       	ndim =2


        connect = {}
        connect['RSS']  = 0
        connect['TOA']  = 0
        connect['TDOA'] = 1
        

        filename = 'Geo_test'
	
        an =zeros(3)

        # node involved in localization
        l_connect=[]

        if connect['RSS']:
                BS1 = np.array([2*L/3.0,0.0,H1])
                l_connect.append('RSS')
                an = vstack((an,BS1))
                BS2 = np.array([L,2*L/3.0,H2])
                l_connect.append('RSS')
                an = vstack((an,BS2))
                BS3 = np.array([L/3.0,L,H3])
                l_connect.append('RSS')
                an = vstack((an,BS3))
                BS4 = np.array([0.0,L/3.0,H1])
                l_connect.append('RSS')
                an = vstack((an,BS4))
                filename = filename + '_RSSI'

        

        if connect['TOA']:
                MS1 = np.array([L/3.0,0.0,H1])
                l_connect.append('TOA')
		an = vstack((an,MS1))
                MS2 = np.array([L,L/3.0,H2])
                l_connect.append('TOA')
		an = vstack((an,MS2))
                MS3 = np.array([2*L/3.0,L,H2])
                l_connect.append('TOA')
		an = vstack((an,MS3))
                MS4 = np.array([0.0,2*L/3.0,H3])
                l_connect.append('TOA')
                an = vstack((an,MS4))
                filename = filename + '_TOA'


        if connect['TDOA']:
                Ap1 = np.array([0.0,0.0,H1])
                l_connect.append('TDOA')
                an = vstack((an,Ap1))
                Ap2 = np.array([L,0.0,H2])
                l_connect.append('TDOA')
                an = vstack((an,Ap2))
                Ap3 = np.array([0.0,L,H3])
                l_connect.append('TDOA')
		an = vstack((an,Ap3))
                Ap4 = np.array([L,L,H1])
                l_connect.append('TDOA')
                an = vstack((an,Ap4))

                filename = filename + '_TDOA'
        

        an=np.delete(an,0,0)

        filename = filename + '.pdf'
        
        ##### LIMITE POUR CONTRAINTE EXCLUDE
	
        BOUND1 = np.array([0,0,-0.05])
        BOUND2 = np.array([L,L,0.05])
  	
	if ndim == 2:
		BOUND1=BOUND1[:2]
		BOUND2=BOUND2[:2]
        ##### RANDOM BLIND NODES
	#random.seed(seed=0)
        bn   = L*rand(Nbn,ndim)
	if ndim == 3:
	        bn[:,2]   = 0.0 




        ############ SCENARION PARAMETERS
        parmsc = {}
        parmsc['room']=vstack((np.min(bn[:,:2],axis=0),np.max(bn[:,:2],axis=0)))
        parmsc['algebraic']      = True
        parmsc['CRB']      = True 

        parmsc['exclude_out']    = np.array((BOUND1 ,BOUND2)) # contraint boundary or None
        parmsc['vcw']            = 1.0
        parmsc['save_pe']        = True 
        parmsc['save_CLA']       = False 

        parmsc['sigma_max_RSS']      = 3.98        # en (dB) 
        parmsc['sigma_max_TOA']      = 1.142       # en (m)  
        parmsc['sigma_max_TDOA']      = 1.85      # en (m)  

        parmsc['std_dev_arange'] = np.array([3]) # arange(1.,4.,1.)
        parmsc['scenar_type']    ='simulated'    # 'monte_carlo'  # choose the type of scenario : Simulated : give some AN and BN. 
        parmsc['err_type']       ='hybrid'         # homogene/hybrid = error applied on AN
        parmsc['Constrain_Type'] ='hybrid'          # 'TOA', 'RSS'
        parmsc['l_connect']      = l_connect
        ## TDOA parmsc
        parmsc['Algebraic_method']    ='ML'          # 'TLS'
        ## RSS parmsc
        parmsc['rss_mode']       = 'mode'
        parmsc['rss0']           = -34.7
        parmsc['d0']             = 1.0
        parmsc['pn']             = 2.64
        parmsc['an_nb']          =  len(an)      # choose the number of AN for performing positionning
        # Monte Carlo simulation parameters     
        parmsc['eval_pos']       = True          # if True , the position of the last evaluated layer is estimated
        parmsc['ceval']          = True          # if True continuous evaluation



        ######################" CLA.SHOW3 PARAMETERS
        parmsh = {}
        parmsh['display']=False     # launch geomview interactively 
        parmsh['scene']=True      # display whole scene
        parmsh['boxes']=True       # display constraint box
        parmsh['constr_boxes']=True       # display constraint box
        parmsh['estimated']=True  # display estimated point

        #
        #  Create the scenario 
        #
	if ndim == 2:
		an =an[:,:ndim]
        S = Scenario(an,bn[:,:ndim],parmsc,parmsh)

	a=time.time()
        S.compute()
	print 'RGPA',S.err1,'in a mean of ', np.mean(S.trgpa),'sec'
	print 'ALG',S.errLS,'in a mean of', np.mean(S.talg),'sec'


	Res=np.array(([np.mean(S.trgpa),np.mean(S.talg)]))
#	np.save(str(connect),Res)


	S.show3()
#	print S.cla.dlayer[2][0].lbnum

#	plt.clf()
#	plt.ion()
#	plt.plot(unique(S.cla.dlayer[S.cla.Nc][0].bd[:,0]))
#	plt.plot(unique(S.cla.dlayer[S.cla.Nc][0].bd[:,1]))
#	plt.plot(unique(S.cla.dlayer[S.cla.Nc][1].bd[:,0]))
#	plt.plot(unique(S.cla.dlayer[S.cla.Nc][1].bd[:,1]))
#	plt.show()


