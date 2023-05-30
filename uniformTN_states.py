#!/usr/bin/env python# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg
from ncon import ncon
from einsumt import einsumt as einsum
from abc import ABC, abstractmethod

from uniformTN_transfers import *
from uniformTN_gradients import *
from uniformTN_functions import polarDecomp,randoUnitary,project_mpsTangentVector,project_mpoTangentVector

class stateAnsatz(ABC):
    @abstractmethod
    def randoInit(self):
        pass

    @abstractmethod
    def shiftTensors(self):
        pass
    @abstractmethod
    def norm(self):
        pass

    def gradDescent(self,H,learningRate,TDVP=False):
        from uniformTN_gradFactory import gradFactory
        gradEvaluater = gradFactory(self,H)
        gradEvaluater.eval()
        if TDVP is True:
            gradEvaluater.projectTDVP()
        self.shiftTensors(-learningRate,gradEvaluater.grad)
        self.norm()

    def gaugeTDVP(self):
        pass

    @abstractmethod
    def get_transfers(self):
        pass
    @abstractmethod
    def get_fixedPoints(self):
        pass
    @abstractmethod
    def get_inverses(self):
        pass
    @abstractmethod
    def del_transfers(self):
        pass
    @abstractmethod
    def del_fixedPoints(self):
        pass
    @abstractmethod
    def del_inverses(self):
        pass
    
class uMPS_1d(stateAnsatz):
    def __init__(self,D,mps=None):
        self.mps = mps
        self.D = D
    def randoInit(self):
        #WLOG just initialise as left canonical (always possible due to gauge)
        self.mps = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    def get_transfers(self):
        self.Ta = mpsTransfer(self.mps)
    def get_fixedPoints(self):
        self.R = self.Ta.findRightEig()
        self.L = self.Ta.findLeftEig()
        self.R.norm_pairedVector(self.L.vector)
    def get_inverses(self):
        self.Ta_inv = inverseTransfer(self.Ta,self.L.vector,self.R.vector)
    def del_transfers(self):
        del self.Ta
    def del_fixedPoints(self):
        del self.R
        del self.L
    def del_inverses(self):
        del self.Ta_inv
    def shiftTensors(self,coef,tensor):
        self.mps += coef*tensor
    def norm(self):
        T = mpsTransfer(self.mps)
        e,u = np.linalg.eig(T.matrix)
        e0 = e[np.argmax(np.abs(e))]
        self.mps = self.mps / np.sqrt(np.abs(e0))

class uMPS_1d_left(uMPS_1d):
    def randoInit(self):
        self.mps = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    def get_fixedPoints(self):
        self.R = self.Ta.findRightEig()
        self.R.norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = inverseTransfer_left(self.Ta,self.R.vector)
    def del_fixedPoints(self):
        del self.R
    def norm(self):
        self.mps = polarDecomp(self.mps.reshape(2*self.D,self.D)).reshape(2,self.D,self.D)

class uMPS_1d_left_twoSite(uMPS_1d_left):
    def randoInit(self):
        self.mps = randoUnitary(4*self.D,self.D).reshape(2,2,self.D,self.D)
    def get_transfers(self):
        self.Ta = mpsTransfer_twoSite(self.mps)
    def norm(self):
        self.mps = polarDecomp(self.mps.reshape(4*self.D,self.D)).reshape(2,2,self.D,self.D)

class uMPS_1d_left_bipartite(uMPS_1d_left):
    def randoInit(self):
        self.mps = dict()
        self.mps[1] = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
        self.mps[2] = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    def get_transfers(self):
        self.Ta = dict()
        self.Ta[1] = mpsTransferBip(self.mps[1],self.mps[2])
        self.Ta[2] = mpsTransferBip(self.mps[2],self.mps[1])
    def get_fixedPoints(self):
        self.R = dict()
        for n in range(1,len(self.Ta)+1):
            self.R[n] = self.Ta[n].findRightEig()
            self.R[n].norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = dict()
        self.Ta_inv[1] = inverseTransfer_left(self.Ta[1],self.R[1].vector)
        self.Ta_inv[2] = inverseTransfer_left(self.Ta[2],self.R[2].vector)
    def shiftTensors(self,coef,tensorDict):
        self.mps[1] += coef*tensorDict[1]
        self.mps[2] += coef*tensorDict[2]
    def norm(self):
        self.mps[1] = polarDecomp(self.mps[1].reshape(2*self.D,self.D)).reshape(2,self.D,self.D)
        self.mps[2] = polarDecomp(self.mps[2].reshape(2*self.D,self.D)).reshape(2,self.D,self.D)

class uMPSU_2d(stateAnsatz):
    def __init__(self,D_mps,D_mpo,mps=None,mpo=None):
        self.mps= mps
        self.mpo= mpo
        self.D_mps = D_mps
        self.D_mpo = D_mpo
        
class uMPSU1_2d_left(uMPSU_2d):
    def randoInit(self):
        #random left canon mps tensor
        self.mps = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        #random unitary mpo, left canon
        self.mpo = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))
    def get_transfers(self):
        self.Ta = mpsTransfer(self.mps)
        T = self.Ta.findRightEig()
        T.norm_pairedCanon()
        self.Tb = mpsu1Transfer_left_oneLayer(self.mps,self.mpo,T)
        self.Tb2 = mpsu1Transfer_left_twoLayer(self.mps,self.mpo,T)
    def get_fixedPoints(self):
        self.T = self.Ta.findRightEig()
        self.R = self.Tb.findRightEig()
        self.RR = self.Tb2.findRightEig()
        self.T.norm_pairedCanon()
        self.R.norm_pairedCanon()
        self.RR.norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = inverseTransfer_left(self.Ta,self.T.vector)
        self.Tb_inv = inverseTransfer_left(self.Tb,self.R.vector)
        self.Tb2_inv = inverseTransfer_left(self.Tb2,self.RR.vector)
        self.Ta_inv.genInverse()
        self.Ta_inv.genTensor()
    def del_transfers(self):
        del self.Ta
        del self.Tb
        del self.Tb2
    def del_fixedPoints(self):
        del self.T
        del self.R
        del self.RR
    def del_inverses(self):
        del self.Ta_inv
        del self.Tb_inv
        del self.Tb2_inv
    def gaugeTDVP(self):
        Ta = mpsTransfer(self.mps)
        T = Ta.findRightEig()
        #gauge transform so diagonal in Z basis
        rho = np.einsum('ija,ujb,ba->ui',self.mps,self.mps.conj(),T.tensor)
        e,u =np.linalg.eig(rho)
        #now gauge transform so diagonal along diagonal in YZ plane
        theta = -math.pi/4
        u2 = np.array([[np.cos(theta/2),1j*np.sin(theta/2)],[1j*np.sin(theta/2),np.cos(theta/2)]])
        u = np.dot(u,u2)
        self.mps = np.einsum('ijk,ia->ajk',self.mps,u)
        self.mpo = np.einsum('ijab,kj->ikab',self.mpo,u.conj().transpose())
    def norm(self):
        #polar decomp to ensure left canon still
        self.mps = polarDecomp(self.mps.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))
    def shiftTensors(self,coef,tensorDict):
        self.mps += coef*tensorDict['mps']
        self.mpo += coef*tensorDict['mpo']

class uMPSU1_2d_left_bipartite(uMPSU_2d):
    def randoInit(self):
        self.mps1 = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        self.mps2 = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        self.mpo1 = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))
        self.mpo2 = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))
    def get_transfers(self):
        self.Ta_12 = mpsTransferBip(self.mps1,self.mps2)
        self.Ta_21 = mpsTransferBip(self.mps2,self.mps1)
        T1 = self.Ta_12.findRightEig()
        T2 = self.Ta_21.findRightEig()
        T1.norm_pairedCanon()
        T2.norm_pairedCanon()
        self.Tb_12 = mpsu1Transfer_left_oneLayerBip(self.mps1,self.mps2,self.mpo1,self.mpo2,T1,T2)
        self.Tb_21 = mpsu1Transfer_left_oneLayerBip(self.mps2,self.mps1,self.mpo2,self.mpo1,T2,T1)
        self.Tb2_12 = mpsu1Transfer_left_twoLayerBip(self.mps1,self.mps2,self.mpo1,self.mpo2,T1,T2)
        self.Tb2_21 = mpsu1Transfer_left_twoLayerBip(self.mps2,self.mps1,self.mpo2,self.mpo1,T2,T1)
    def get_fixedPoints(self):
        self.T1 = self.Ta_12.findRightEig()
        self.T2 = self.Ta_21.findRightEig()
        self.T1.norm_pairedCanon()
        self.T2.norm_pairedCanon()
        self.R1 = self.Tb_12.findRightEig()
        self.R2 = self.Tb_21.findRightEig()
        self.R1.norm_pairedCanon()
        self.R2.norm_pairedCanon()
        self.RR1 = self.Tb2_12.findRightEig()
        self.RR2 = self.Tb2_21.findRightEig()
        self.RR1.norm_pairedCanon()
        self.RR2.norm_pairedCanon()
    def get_inverses(self):
        self.Ta_12_inv = inverseTransfer_left(self.Ta_12,self.T1.vector)
        self.Ta_21_inv = inverseTransfer_left(self.Ta_21,self.T2.vector)
        self.Tb_12_inv = inverseTransfer_left(self.Tb_12,self.R1.vector)
        self.Tb_21_inv = inverseTransfer_left(self.Tb_21,self.R2.vector)
        self.Tb2_12_inv = inverseTransfer_left(self.Tb2_12,self.RR1.vector)
        self.Tb2_21_inv = inverseTransfer_left(self.Tb2_21,self.RR2.vector)
    def del_transfers(self):
        del self.Ta_12 
        del self.Ta_21 
        del self.Tb_12 
        del self.Tb_21 
        del self.Tb2_12 
        del self.Tb2_21 
    def del_fixedPoints(self):
        del self.T1 
        del self.T2 
        del self.R1 
        del self.R2 
        del self.RR1 
        del self.RR2 
    def del_inverses(self):
        del self.Ta_12_inv 
        del self.Ta_21_inv 
        del self.Tb_12_inv 
        del self.Tb_21_inv 
        del self.Tb2_12_inv 
        del self.Tb2_21_inv 
    def gaugeTDVP(self):
        Ta_12 = mpsTransferBip(self.mps1,self.mps2)
        Ta_21 = mpsTransferBip(self.mps2,self.mps1)
        T1 = Ta_12.findRightEig()
        T2 = Ta_21.findRightEig()
        T1.norm_pairedCanon()
        T2.norm_pairedCanon()

        #gauge transform so diagonal along diagonal in YZ plane
        theta = math.pi/4
        u2 = np.array([[np.cos(theta/2),1j*np.sin(theta/2)],[1j*np.sin(theta/2),np.cos(theta/2)]])

        #gauge transform so diagonal in Z basis
        rho = np.einsum('ija,ujb,ba->ui',self.mps1,self.mps1.conj(),T2.tensor)
        e,u =np.linalg.eig(rho)
        u = np.dot(u,u2)
        self.mps1 = np.einsum('ijk,ia->ajk',self.mps1,u)
        self.mpo1 = np.einsum('ijab,kj->ikab',self.mpo1,u.conj().transpose())

        #gauge transform so diagonal in Z basis
        rho = np.einsum('ija,ujb,ba->ui',self.mps2,self.mps2.conj(),T1.tensor)
        e,u =np.linalg.eig(rho)
        u = np.dot(u,u2)
        self.mps2 = np.einsum('ijk,ia->ajk',self.mps2,u)
        self.mpo2 = np.einsum('ijab,kj->ikab',self.mpo2,u.conj().transpose())
    def shiftTensors(self,coef,tensorDict):
        self.mps1 += coef*tensorDict['mps1']
        self.mps2 += coef*tensorDict['mps2']
        self.mpo1 += coef*tensorDict['mpo1']
        self.mpo2 += coef*tensorDict['mpo2']
    def norm(self):
        #polar decomps to project to closest unitaries
        self.mps1 = polarDecomp(self.mps1.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        self.mps2 = polarDecomp(self.mps2.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        self.mpo1 = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo1).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))
        self.mpo2 = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo2).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))

class uMPS_1d_centre(uMPS_1d_left):
    def randoInit(self):
        super().randoInit() #initialise as random uniform mps, left canon
        self.construct_centreGauge_tensors()

    def construct_centreGauge_tensors(self):
        #get left/right dom eigvectors L,R
        T = mpsTransfer(self.mps)
        R = T.findRightEig()
        R.norm_pairedCanon()

        #decompose L=l^+l, R=rr^+
        l = np.eye(self.D)
        U,S,Vd = sp.linalg.svd(R.tensor,full_matrices=False)
        r = np.dot(U,np.diag(np.sqrt(S)))
        #mixed canon form
        C = np.dot(l,r)
        U,S,Vd = sp.linalg.svd(C)
        Ac = ncon([self.mps,l,r],((-1,3,4),(-2,3),(4,-5)),forder=(-1,-2,-5))
        self.Ac = Ac
        self.C = C
        #find Ar,Al
        self.polarDecomp_bestCanon()

    def polarDecomp_bestCanon(self):
        Ac_l = self.Ac.reshape(2*self.D,self.D)
        Ac_r = np.einsum('ijk->jik',self.Ac).reshape(self.D,2*self.D)

        Al_notCanon = np.dot(Ac_l,self.C.conj().transpose())
        Ar_notCanon = np.dot(self.C.conj().transpose(),Ac_r)
        Ul,Sl,Vl = sp.linalg.svd(Al_notCanon,full_matrices=False)
        Ur,Sr,Vr = sp.linalg.svd(Ar_notCanon,full_matrices=False)
        self.Al = np.dot(Ul,Vl).reshape(2,self.D,self.D)
        self.Ar= np.einsum('jik->ijk',np.dot(Ur,Vr).reshape(self.D,2,self.D))

    def polarDecomp_bestCanon_stable(self):
        Ac_l = self.Ac.reshape(2*self.D,self.D)
        Ac_r = np.einsum('ijk->jik',self.Ac).reshape(self.D,2*self.D)
        U_ac_l,S_ac_l,Vd_ac_l = sp.linalg.svd(Ac_l,full_matrices=False)
        U_ac_l = np.dot(U_ac_l,Vd_ac_l)

        U_ac_r,S_ac_r,Vd_ac_r = sp.linalg.svd(Ac_r,full_matrices = False)
        U_ac_r = np.dot(U_ac_r,Vd_ac_r)

        Uc,Sc,Vdc = sp.linalg.svd(self.C,full_matrices = False)
        Uc_l = np.dot(Uc,Vdc)
        Uc_r = np.dot(Uc,Vdc)

        self.Al = np.dot(U_ac_l,Uc_l.conj().transpose()).reshape(2,self.D,self.D)
        self.Ar = np.einsum('jik->ijk',np.dot(Uc_r.conj().transpose(),U_ac_r).reshape(self.D,2,self.D))

    def vumps_update(self,beta,twoBodyH,H_ac,H_c,stable_polar=True):
        #regular eigensolver #replace with lanzos for bigger D
        e_ac,u_ac = sp.linalg.eigh(H_ac)
        e_c,u_c = sp.linalg.eigh(H_c)
        #imaginary time evolve to pick out gs with component along previous vector 
        #(like power method to deal with degeneracies, but only one param beta) 
        Ac0 = np.dot(u_ac.conj().transpose(),self.Ac.reshape(2*self.D**2))
        C0 = np.dot(u_c.conj().transpose(),self.C.reshape(self.D**2))
        Ac = np.multiply(Ac0,np.exp(-beta*(e_ac-e_ac[np.argmin(e_ac)]))) #shift spectrum so gs is E=0, avoids nans
        C = np.multiply(C0,np.exp(-beta*(e_c-e_c[np.argmin(e_c)])))
        Ac = Ac / np.power(np.vdot(Ac,Ac),0.5)
        C = C / np.power(np.vdot(C,C),0.5)
        self.Ac = np.dot(u_ac,Ac).reshape(2,self.D,self.D)
        self.C = np.dot(u_c,C).reshape(self.D,self.D)
        if stable_polar is True:
            self.polarDecomp_bestCanon_stable()
        else:
            self.polarDecomp_bestCanon()
        self.mps = self.Al
