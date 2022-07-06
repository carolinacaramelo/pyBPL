#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:12:54 2022

@author: carolinacaramelo
"""

import matplotlib.pyplot as plt
import numpy as np
from pybpl.library import Library
from pybpl.model import CharacterModel
from pybpl.model  import type_dist
from pybpl.model.type_dist import ConceptTypeDist, CharacterTypeDist, PartTypeDist, StrokeTypeDist
from pybpl.objects import StrokeType, ConceptType, CharacterType
import torch
import torch.distributions as dist

from pybpl.library import Library
from pybpl.model.type_dist import ConceptTypeDist, CharacterTypeDist, PartTypeDist, StrokeTypeDist, RelationTypeDist



class Perturbing (StrokeTypeDist, ConceptTypeDist, CharacterModel, Library):
    
    def __init__ (self):

    
        # initializaton of class inheritances
        self.lib = Library(use_hist = True)
        self.STD = StrokeTypeDist(self.lib)
        self.CTD = ConceptTypeDist(self.lib)
        self.CM = CharacterModel(self.lib)
        self.RTD = RelationTypeDist(self.lib)
        
        
        
        
        # Generative process relevant variables 
        #self.ids = torch.tensor([1]) # ids of the substrokes
        self.n_strokes = 7 # number of strokes
        self.subparts = [30,300,45,70,89] #possible ids to use when sampling the id list 
        self.alpha = 0 
        self.nsub = torch.tensor ([1])
        
        
    
    
    # add to StrokeTypeDist class
    def get_part_type (self):
        """
        Define a stroke type from a list of sub-stroke ids
        Parameters
        ----------
        ids : tensor
            vector, list of sub-part/stroke indices
        Returns
        -------
        p : StrokeType
            part type sample (strokes are the same things as parts)
        """
        # sample the number of sub-strokes
        nsub = self.ids.shape
        # sample control points for each sub-stroke in the sequence
    

        # sample control points for each sub-stroke in the sequence
        shapes = self.STD.sample_shapes_type(self.ids)
        # sample scales for each sub-stroke in the sequence
        invscales = self.STD.sample_invscales_type(self.ids)
        p = StrokeType(nsub, self.ids, shapes, invscales)
        
        return p 
       
    
# add to ConceptTypeDist class?
    def known_sample_type (self):
    # initialize relation type lists
        self.R = []
        self.P = []
        # for each part, sample part parameters
        for _ in range(self.n_strokes):
            # sample the relation type
            r = self.CTD.rdist.sample_relation_type(self.P)
            p= self.get_part_type()
            # append to the lists
            self.R.append(r)
            self.P.append (p)
 
        
    

    #add to CharacterTypeDist class
    def known_ctype (self):
        
        self.known_sample_type()
        print(self.R, self.P)
        CT = ConceptType(self.n_strokes, self.P, self.R)
       
        #c = super(CharacterTypeDist, self).sample_type(self.n_strokes)
        #ctype = CharacterType(c.k, c.part_types, c.relation_types)

        return CharacterType(self.n_strokes, CT.part_types, CT.relation_types)
    
    
    #add to CharacterModel class
    def known_stype(self):
         return self.known_ctype()
     
        
   # def display_type(self,c_type):
       # print('----BEGIN CHARACTER TYPE INFO----')
       # print('num strokes: %i' % self.n_strokes)
       # for i in range(self.n_strokes):
           # print('Stroke #%i:' % i)
           # print('\tsub-stroke ids: ', list(self.ids)
            #print('\trelation category: %s' % self.RTD.relation_types[i].category)
        #print('----END CHARACTER TYPE INFO----')
     
  
    def run_example(self,n_strokes, nsub, alpha, n_examples):
        self.n_strokes = n_strokes # number of strokes
        subpart_sample = np.linspace(0,1212,1212, dtype = int)
        subp_list = [np.random.choice(subpart_sample),np.random.choice(subpart_sample)] 
        print(subp_list)
        self.subparts = subp_list
        self.alpha = alpha 
        self.nsub = torch.tensor ([nsub])
        fig, axes = plt.subplots(nrows=10, ncols=n_examples, figsize=(10, 50))
        #if perturbed:
        for i in range(10):
            self.sample_ids(False)
            self.get_part_type()
            c_type = self.known_stype() #list of character types 
            print('type %i' % i)
            #display_type(ctype)
            # sample a few character tokens and visualize them
            for j in range(n_examples):
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                axes[i,j].imshow(c_image, cmap='Greys')
                axes[i,j].tick_params(
                    which='both',
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False
                )
            axes[i,0].set_ylabel('%i' % i, fontsize=50)
        print("Non perturbed")
        plt.show()
        #else:
            
        fig, axes = plt.subplots(nrows=10, ncols=n_examples, figsize=(10, 50))
        for i in range(10):
            self.sample_ids(True)
            self.get_part_type()
            c_type = self.known_stype() #list of character types 
            print('type %i' % i)
            # sample a few character tokens and visualize them
            for j in range(n_examples):
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                axes[i,j].imshow(c_image, cmap='Greys')
                axes[i,j].tick_params(
                    which='both',
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False
                )
            axes[i,0].set_ylabel('%i' % i, fontsize=50)
        print("Perturbed")
        plt.show()


                
                

        
    #remove entries that don't correspond to the possible indexes
    #def truncate_start (self, subparts):
        #s = self.lib.logStart
        #m = np.isin([i for i in range(len(s))], subparts)
        #m = np.invert(m)
        #indices = np.argwhere(m)
        #s[indices] = 0
    
        #return s
        
        
        
   #remove entries that don't correspond to the possible indexes
    def truncate_start (self, pT):
        m = np.isin([i for i in range(len(pT))], self.subparts)
        m = np.invert(m)
        indices = np.argwhere(m)
        pT[indices] = 0
    
        return pT
    
    #remove entries that don't correspond to the possible indexes
    #def truncateT (self, subparts):
       #p = self.lib.logT
       #m = np.isin([i for i in range(len(p))], subparts)
       #n = np.invert(m)
       #indices2 = np.argwhere(n)
       #p[:,indices2]=0
       #p[indices2,:]=0
       #return p
       
   #remove entries that don't correspond to the possible indexes
    def truncateT (self):
        logR = self.lib.logT
        R = torch.exp(logR)
        m = np.isin([i for i in range(len(R))], self.subparts)
        n = np.invert(m)
        indices2 = np.argwhere(n)
        R[:,indices2]=0
        R[indices2,:]=0
        #print (R)
        return R
    
    def truncateT_perturbed (self): #in gaussian noise the alpha can be sqrt(variance) #perturbar antes a matrix logT, perturbar sem normalizar na logT 
        logR = self.lib.logT
        s = list(logR.size())
        M_perturbation = torch.randn(s[0], s[1]) #try gaussian first #The function torch.randn produces a tensor with elements drawn from a Gaussian distribution of zero mean and unit variance. Multiply by sqrt(0.1) to have the desired variance.
        logR_p = logR + self.alpha * M_perturbation 
        R = torch.exp(logR_p)
        m = np.isin([i for i in range(len(R))], self.subparts)
        n = np.invert(m)
        indices2 = np.argwhere(n)
        R[:,indices2]=0
        R[indices2,:]=0
        #print (R)
        return R
        
    
    def pT_truncate (self,ids):
        #assert ids.shape == torch.Size([])
        R = self.truncateT ()
        R = R[ids]
        #print(R)
        pT = R / torch.sum(R)
        return pT

        
    def pT_perturbed (self,ids): 
        #assert ids.shape == torch.size ([])
        R = self.truncateT_perturbed()
        R = R[ids]
        #print(R)
        pT_perturbed = R / torch.sum(R)
        return pT_perturbed
        
        
    #subparts are the available ids, nsub are the number of ids we want to sample
    def sample_ids (self, perturbed): #perturbed : bool, alpha
        #set the initial transition probabilities 
        pT = torch.exp(self.lib.logStart)
        pT = self.truncate_start (pT)
        print(pT)
        # sub-stroke sequence is a list, this is the list of ids that we will sample
        ids =[]
        if perturbed:
            # step through and sample 'nsub' sub-strokes
            for _ in range(self.nsub):
                ss = torch.multinomial(pT,1)
                ss= torch.squeeze (ss,-1)
                ids.append(ss)
                # update transition probabilities; condition on previous sub-stroke
                pT = self.pT_perturbed(ss)
        else: 
            for _ in range(self.nsub):
                ss = torch.multinomial(pT,1)
                ss= torch.squeeze (ss,-1)
                ids.append(ss)
                print (ids)
                # update transition probabilities; condition on previous sub-stroke
                pT = self.pT_truncate(ss)
                print (pT)
        
        # convert list into tensor
        self.ids = torch.stack(ids)
        print(ids)
        return ids
    
        
            
       
#o objetivo é dar uma lista de ids possíveis a utilizar e a partir daí fazer o sampling da ordem em que os ids aparecem usando pT 
# ou então a pT perturbada, por isso o que vamos ter é uma função que recebe uma lista de ids e usa pT, outra função que recebe a 
# a lista de ids e usa a pT perturbed, e uma função que recebe como argumento a lista restrita de ids e diz se perturbamos ou não 
# o que esta última função vai fazer é cuspir uma lista desses ids ordenados de maneira diferente - depois corremos o exemplo com
# essa lista de ids e vemos a diferença entre a imagem gerada com pT ou com pT perturbed 
       







        
