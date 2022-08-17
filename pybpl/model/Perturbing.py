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
from pybpl.objects import StrokeType, ConceptType, CharacterType
import torch
import torch.distributions as dist
import os 
from PIL import Image as img
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
        #dcan be previously defined 
        #self.ids = torch.tensor([1]) # ids of the substrokes
        self.n_strokes = 7 # number of strokes, it has to be <=10
        self.subparts = [30,300,45,70,89] #possible ids to use when sampling the id list 
        self.nsub = torch.tensor ([1]) #number of subparts its also <=10, in the normal case it depends on n_strokes
        
        self.seq_ids = [[]]
        
    
    
    #altered function of StrokeTypeDist class, defines a stroke type 
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
        # get the number of sub-strokes
        nsub = self.ids.shape
        # sample control points for each sub-stroke in the sequence
    

        # sample control points for each sub-stroke in the sequence
        shapes = self.STD.sample_shapes_type(self.ids)
        # sample scales for each sub-stroke in the sequence
        invscales = self.STD.sample_invscales_type(self.ids)
        p = StrokeType(nsub, self.ids, shapes, invscales)
        
        return p 
       
    
    #altered function from ConceptTypeDist class, defines a concept type
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
        ctype = ConceptType(self.n_strokes, self.P, self.R)

        return ctype
 
        
    

    #altered function from CharacterTypeDist class, defines a character type
    def known_ctype (self):
        
        self.known_sample_type()
        print(self.R, self.P)
        CT = ConceptType(self.n_strokes, self.P, self.R)
       
        #c = super(CharacterTypeDist, self).sample_type(self.n_strokes)
        #ctype = CharacterType(c.k, c.part_types, c.relation_types)

        return CharacterType(self.n_strokes, CT.part_types, CT.relation_types)
    
    
    #aaltered function from CharacterModel class, sampling the character type of the image (model) that will be generated
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
     
   
    
   #running individual examples   
    def run_example_random(self,n_strokes, nsub, alpha, n_examples):
        self.n_strokes = n_strokes # number of strokes, we decide the number of strokes
        subpart_sample = np.linspace(0,1212,1212, dtype = int) #define the space of the subparts
        subp_list = [np.random.choice(subpart_sample),np.random.choice(subpart_sample)] #here we are just sampling two random subparts 
        print(subp_list)
        self.subparts = subp_list #possible ids to use when sampling the id list 
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

    
   
    
   
    #generate alphabet will generate a set of character images related to each other - one alphabet 
    #the function will generate an alphabet with the unperturbed model and one alphabet for the perturbed model
    #the generation of the alphabets for both cases will use the same number of strokes, same number of sub-strokes and the the same primitives 
    #each alphabet will have 30 characters 
    #the sub-strokes/primitives used will be transformed into images and stores in order to have a better comprehension of what these are 
    #the alphabets will be stored in a folder 
    #20 tokens of each one of the character types 
    #1 alphabet - 30 character types - 20 tokens for each character type
    def generate_alphabet(self, n_strokes, nsub, alpha, n_characters, n_tokens):
        #set the parameters to sample the primitives 
        self.n_strokes = 1 #we begin 1 stroke and 1 subtroke in order to visualize the primitives that are being used 
        self.nsub = torch.tensor ([1])
        subpart_idx = np.linspace(1,1212,1212, dtype = int) #creates an a array of the primitive indexes 1-1212
        subpart_list = []
        path = '/Users/carolinacaramelo/Desktop/alphabets'
        os.makedirs(path)
        for i in range(nsub): #print the primitives we are using, being able to visualize what are the sampled primitives
       
            subpart_sample = np.random.choice(subpart_idx)
            print (subpart_sample ,'olá')
            subpart_list += [subpart_sample]
            self.subparts = subpart_sample
            self.sample_ids(False) 
            self.get_part_type()
            c_type = self.known_stype()
            c_token = self.CM.sample_token(c_type)
            c_image = self.CM.sample_image(c_token)
            plt.rcParams["figure.figsize"] = (10,50)
            plt.subplot(nsub,1,i+1)
            plt.imshow(c_image, cmap="Greys")
            print(subpart_list , 'adeus')
            
        
        plt.savefig('primitives.png')
        primitives = img.open (r"/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pybpl/model/primitives.png") 
        primitives.save('/Users/carolinacaramelo/Desktop/alphabets/primitives.png', 'PNG')
        
        #set the parameters to sample the alphabet
        #instead of giving the n_strokes and nsub we could also randomly sample these numbers and generate more than one alphabet all at once (with different n_strokes and nsub)
        self.n_strokes = n_strokes #know we are using these number of strokes and substrokes
        self.nsub = torch.tensor ([nsub]) #we are going to use the same number of strokes and substrokes for both alphabets
        self.subparts = subpart_list
        self.alpha = alpha #for the perturbation
        self.n_characters = n_characters #for the new logT function
        
        seq_list=[]
        #generate non perturbed alphabet 
        path = '/Users/carolinacaramelo/Desktop/alphabets/nonperturbed'
        os.makedirs(path)
        for i in range(n_characters):
            
            path_character = '/Users/carolinacaramelo/Desktop/alphabets/nonperturbed/character%i'%i
            os.makedirs(path_character)
            self.sample_ids(False)
            seq_list[i] = self.seq #store the sequence of indexes 
            self.get_part_type()
            c_type = self.known_stype() #list of character types 
            
            #saves different tokens of the same character
            for j in range(n_tokens):
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                plt.rcParams["figure.figsize"] = (10,50)
                plt.subplot(1,1,1)
                plt.imshow(c_image, cmap="Greys")
                plt.savefig('char%s.' %j +'.png')
                char = img.open (r"/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pybpl/model/char%s." %j+".png") 
                char.save('/Users/carolinacaramelo/Desktop/alphabets/nonperturbed/character%i' %i + '/char%s.' %j, 'PNG')
        
        #generate perturbed alphabet 
        path = '/Users/carolinacaramelo/Desktop/alphabets/perturbed'
        os.makedirs(path)
        for i in range(n_characters):
            
            path_character = '/Users/carolinacaramelo/Desktop/alphabets/perturbed/character%i'%i
            os.makedirs(path_character)
            self.sample_ids(True) 
            seq_list[n_characters+1+i] = self.seq #continue storing the sequence of indexes
            self.get_part_type()
            c_type = self.known_stype() #list of character types 
            
            #saves different tokens of the same character
            for j in range(n_tokens):
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                plt.rcParams["figure.figsize"] = (10,50)
                plt.subplot(1,1,1)
                plt.imshow(c_image, cmap="Greys")
                plt.savefig('char%s.' %j +'.png')
                char = img.open (r"/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pybpl/model/char%s." %j+".png") 
                char.save('/Users/carolinacaramelo/Desktop/alphabets/perturbed/character%i' %i + '/char%s.' %j, 'PNG')
        self.seq_ids = seq_list
        
        
   #empiracally count the number of transitions that happen between primitives 
   #save the lists of primitive indexes' sequences 
   #know the number of symbols that were generated 
   #check every combination of transitions - row by row of the transition matrix 
    def new_logT(self):
        t_counts= self.nsub * self.n_characters - self.n_characters #number of total counts - it has to be the number of transitions that exist in seq_ids - still have to count this...
        T = torch.zeros((1212,1212))
        for i in self.subparts:
            print (i)
            counts=0
            for j in self.subparts:
                print(j)
                for l in range(len(self.seq_ids)-1):
                    if i in self.seq_ids[l]:
                        if j in self.seq_ids[l]:
                            if self.seq_ids[l][self.seq_ids[l].index(i)+1] == j: #this is a problem, there can be more than one index where i exists
                                counts += 1
                                print ('counts %s.' %counts)
                p = counts/t_counts 
                print(p)
                #falta acrescentar depois aqui a matriz T e no final fazer log de tudo parwa obtermos logT
                T[i,j]= p
                print(T)
        logT = torch.log(T)
        self.logT = logT
        return logT
        
                        
                    
            
        
        
        
        
        
        
        
        

            
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
    
    def M_perturbation(self,str, alpha):
        logR = self.lib.logT
        s = list(logR.size())
        if str == "Gaussian":
            M_perturbation = (alpha**0.5) * torch.randn(s[0], s[1]) #The function torch.randn produces a tensor with elements drawn from a Gaussian distribution of zero mean and unit variance. Multiply by sqrt(0.1) to have the desired variance.
        #if str == "Dirichlet":
        #if str == "Poisson":
        #if str == "Exponential":
            
            
        return M_perturbation
            
    
   
    def truncateT_perturbed (self): #in gaussian noise the alpha can be sqrt(variance) #perturbar antes a matrix logT, perturbar sem normalizar na logT 
        logR = self.lib.logT
        logR_p = logR + self.M_perturbation("Gaussian",0)
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
        
        
    #subparts are the available ids, nsub are the number of ids we want to sample (the number of subparts in the id list sequence)
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
        self.pT = pT
        self.ids = torch.stack(ids)
        self.seq = ids
        print(ids)
        return ids
    
        
            
       
#o objetivo é dar uma lista de ids possíveis a utilizar e a partir daí fazer o sampling da ordem em que os ids aparecem usando pT 
# ou então a pT perturbada, por isso o que vamos ter é uma função que recebe uma lista de ids e usa pT, outra função que recebe a 
# a lista de ids e usa a pT perturbed, e uma função que recebe como argumento a lista restrita de ids e diz se perturbamos ou não 
# o que esta última função vai fazer é cuspir uma lista desses ids ordenados de maneira diferente - depois corremos o exemplo com
# essa lista de ids e vemos a diferença entre a imagem gerada com pT ou com pT perturbed 
       


#we can also get the scores for the perturbed version of the model 
# we are not sampling nstrokes and nsub from the distributions but we are scoring the values we aree giving these variables from the correspondent distributions 


    def score_n_strokes(self):
        """
        Compute the log-probability of the stroke count under the prior, the difference is that n_strokes is not sampled, but given by us
    
        Parameters
        ----------
        n_strokes : tensor
            scalar; stroke count to score
    
        Returns
        -------
        ll : tensor
            scalar; log-probability of the stroke count
        """
        # check if any values are out of bounds
        if self.n_strokes > len(self.kappa.probs):
            ll = torch.tensor(-float('Inf'))
        else:
            # score points using kappa
            # NOTE: subtract 1 to get 0-indexed samples
            ll = self.kappa.log_prob(self.n_strokes-1)
    
        return ll
    
    def score_nsub(self):
        """
        Compute the log-probability of a sub-stroke count under the prior. The n_strokes and nsub are given by us 

        Parameters
        ----------
        n_strokes : tensor
            scalar; stroke count
        nsub : tensor
            scalar; sub-stroke count

        Returns
        -------
        ll : tensor
            scalar; log-probability of the sub-stroke count
        """
        # nsub should be a scalar
        assert self.nsub.shape == torch.Size([])
        # collect pvec for thisnumber of strokes
        pvec = self.pmat_nsub[self.n_strokes-1]
        # make sure pvec is a vector
        assert len(pvec.shape) == 1
        # score using the categorical distribution
        ll = dist.Categorical(probs=pvec).log_prob(self.nsub-1)

        return ll
    
    def score_subIDs(self):
        """
        Compute the log-probability of a sub-stroke ID sequence under the prior

        Parameters
        ----------
        ids : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        ll : (nsub,) tensor
            scalar; log-probability of the sub-stroke ID sequence
        """
        # set initial transition probabilities
        pT = self.pT #we have to use the probability transition matrix that was used to do the sampling of the ids in sample_ids (?)
        
        # initialize log-prob vector
        ll = torch.zeros(len(self.ids), dtype=torch.float)
        # step through sub-stroke IDs
        for i, ss in enumerate(self.ids):
            # add to log-prob accumulator
            ll[i] = dist.Categorical(probs=pT).log_prob(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = self.pT(ss)

        return ll
    
    
    def score_shapes_type(self, shapes): #stays the same 
        """
        Compute the log-probability of the control points for each sub-stroke
        under the prior

        Parameters
        ----------
        ids : (nsub,) tensor
            sub-stroke ID sequence
        shapes : (ncpt, 2, nsub) tensor
            shapes of bsplines

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        if self.isunif:
            raise NotImplementedError
        # check that subid is a vector
        assert len(self.ids.shape) == 1
        # record vector length
        nsub = len(self.ids)
        assert shapes.shape[-1] == nsub
        # reshape tensor (ncpt, 2, nsub) -> (ncpt*2, nsub)
        shapes = shapes.view(self.ncpt*2,nsub)
        # transpose axes (ncpt*2, nsub) -> (nsub, ncpt*2)
        shapes = shapes.transpose(0,1)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(
            self.shapes_mu[self.ids], self.shapes_Cov[self.ids]
        )
        # score points using the multivariate normal distribution
        ll = mvn.log_prob(shapes)

        return ll
    
    def score_invscales_type(self,  invscales): #also stays the same, we are not changing the way we sample shapes and scale 
        """
        Compute the log-probability of each sub-stroke's scale parameter
        under the prior

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        invscales : (nsub,) tensor
            scale values for each sub-stroke

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        if self.isunif:
            raise NotImplementedError
        # make sure these are vectors
        assert len(invscales.shape) == 1
        assert len(self.ids.shape) == 1
        assert len(invscales) == len(self.ids)
        # create gamma distribution
        gamma = dist.Gamma(self.scales_con[self.ids], self.scales_rate[self.ids])
        # score points using the gamma distribution
        ll = gamma.log_prob(invscales)

        return ll
    
    def score_part_type(self, ptype):#ptype is gotten from the get_part_type function, it is the stroke type 
        """
        Compute the log-probability of the stroke type, conditioned on a
        stroke count, under the prior

        Parameters
        ----------
        
        ptype : StrokeType
            part type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the stroke type
        """
        nsub_score = self.score_nsub(self.n_strokes, ptype.nsub)
        subIDs_scores = self.score_subIDs(ptype.ids)
        shapes_scores = self.score_shapes_type(ptype.ids, ptype.shapes)
        invscales_scores = self.score_invscales_type(ptype.ids, ptype.invscales)
        ll = nsub_score + torch.sum(subIDs_scores) + torch.sum(shapes_scores) \
             + torch.sum(invscales_scores) #sum of scores

        return ll
    
    def score_type(self, ctype): #we get ctype from known_sample_type
        """
        Compute the log-probability of a concept type under the prior
        $P(type) = P(k)*\prod_{i=1}^k [P(S_i)P(R_i|S_{0:i-1})]$

        Parameters
        ----------
        ctype : ConceptType
            concept type to score

        Returns
        -------
        ll : tensor
            scalar; log-probability of the concept type
        """
        assert isinstance(ctype, ConceptType)
        # score the number of parts
        ll = 0.
        ll = ll + self.score_n_strokes(ctype.k)
        # step through and score each part
        for i in range(ctype.k):
            ll = ll + self.pdist.score_part_type(ctype.k, ctype.part_types[i])
            ll = ll + self.rdist.score_relation_type(
                ctype.part_types[:i], ctype.relation_types[i]
            )

        return ll
    
    
    
    
    
    
    





        
