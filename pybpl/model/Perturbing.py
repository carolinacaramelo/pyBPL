#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:12:54 2022

@author: carolinacaramelo
"""

import matplotlib.pyplot as plt
import numpy as np
from pybpl.library import Library
from pybpl.splines import bspline_gen_s
from pybpl.model import CharacterModel
from pybpl.model  import type_dist
from pybpl.model import token_dist
from pybpl.objects import StrokeType, ConceptType, CharacterType, StrokeToken,ConceptToken,CharacterToken
from pybpl.objects import (RelationType, RelationIndependent, RelationAttach,
                       RelationAttachAlong, RelationToken)
import torch
import torch.distributions as dist
import os 
from PIL import Image as img
from pybpl.model.type_dist import ConceptTypeDist, CharacterTypeDist, PartTypeDist, StrokeTypeDist, RelationTypeDist
from pybpl.model.token_dist import ConceptTokenDist, CharacterTokenDist, PartTokenDist, StrokeTokenDist, RelationTokenDist
from scipy import io



class Perturbing (StrokeTypeDist, ConceptTypeDist, CharacterModel, Library):
    
    def __init__ (self):

    
        # initializaton of class inheritances
        self.lib = Library(use_hist = True)
        self.STD = StrokeTypeDist(self.lib)
        self.CTD = ConceptTypeDist(self.lib)
        self.CM = CharacterModel(self.lib)
        self.RTD = RelationTypeDist(self.lib)
        self.STT = StrokeTokenDist(self.lib)
        self.CTT = ConceptTokenDist(self.lib)
        self.RTT = RelationTokenDist(self.lib)
        
        	
        
        
        
        
        # Generative process relevant variables 
        #dcan be previously defined 
        #self.ids = torch.tensor([1]) # ids of the substrokes
        self.n_strokes = 7 # number of strokes, it has to be <=10
        self.subparts = [30,300,45,70,89] #possible ids to use when sampling the id list 
        self.nsub = torch.tensor ([1]) #number of subparts its also <=10, in the normal case it depends on n_strokes
        
        self.seq_ids = [[]]
        
#### FUNCTIONS FOR SAMPLING THE CHARACTER MODEL TYPE ################################################################################################
    
    #altered function of StrokeTypeDist class, defines a stroke type 
    def get_part_type (self, perturbed):
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
        nsub = self.nsub
        # sample the sequence of sub-stroke IDs
        self.sample_ids (perturbed)
        
    

        # sample control points for each sub-stroke in the sequence
        shapes = self.STD.sample_shapes_type(self.ids)
        # sample scales for each sub-stroke in the sequence
        invscales = self.STD.sample_invscales_type(self.ids)
        p = StrokeType(nsub, self.ids, shapes, invscales)
        
        return p 
       
    
    #altered function from ConceptTypeDist class, defines a concept type
    def known_sample_type (self,perturbed):
    # initialize relation type lists
        self.R = []
        self.P = []
        # for each part, sample part parameters
        for _ in range(self.n_strokes):
            # sample the relation type
            r = self.CTD.rdist.sample_relation_type(self.P)
            p= self.get_part_type(perturbed)
            # append to the lists
            self.R.append(r)
            self.P.append (p)
        ctype = ConceptType(self.n_strokes, self.P, self.R)

        return ctype
 

    #altered function from CharacterTypeDist class, defines a character type
    def known_ctype (self, perturbed):
        
        self.known_sample_type(perturbed)
        print(self.R, self.P)
        CT = ConceptType(self.n_strokes, self.P, self.R)
       
        #c = super(CharacterTypeDist, self).sample_type(self.n_strokes)
        #ctype = CharacterType(c.k, c.part_types, c.relation_types)

        return CharacterType(self.n_strokes, CT.part_types, CT.relation_types)
    
    
    #aaltered function from CharacterModel class, sampling the character type of the image (model) that will be generated
    def known_stype(self, perturbed):
         return self.known_ctype(perturbed)
     
        
   # def display_type(self,c_type):
       # print('----BEGIN CHARACTER TYPE INFO----')
       # print('num strokes: %i' % self.n_strokes)
       # for i in range(self.n_strokes):
           # print('Stroke #%i:' % i)
           # print('\tsub-stroke ids: ', list(self.ids)
            #print('\trelation category: %s' % self.RTD.relation_types[i].category)
        #print('----END CHARACTER TYPE INFO----')
     
   
#### PERTURBED SAMPLING #############################################################################################################
        
        
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
       

###### SCORING SAMPLING #########################################################################################################################

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
    

######## OPTIMIZING CHARACTER MODEL TYPE ###################
        
        def optimize_type(self, c, lr, nb_iter, eps, show_examples=True):
            """
            Take a character type and optimize its parameters to maximize the
            likelihood under the prior, using gradient descent

            Parameters
            ----------
            model : CharacterModel
            c : CharacterType
            lr : float
            nb_iter : int
            eps : float
            show_examples : bool

            Returns
            -------
            score_list : list of float

            """
            # round nb_iter to nearest 10
            nb_iter = np.round(nb_iter, -1)
            # get optimizable variables & their bounds
            c.train()
            params = c.parameters()
            lbs = c.lbs(eps)
            ubs = c.ubs(eps)
            # optimize the character type
            score_list = []
            optimizer = torch.optim.Adam(params, lr=lr)
            if show_examples:
                fig, axes = plt.subplots(10, 4, figsize=(4, 10))
            interval = int(nb_iter / 10)
            for idx in range(nb_iter):
                if idx % interval == 0:
                    # print optimization progress
                    print('iteration #%i' % idx)
                    if show_examples:
                        # sample 4 tokens of current type (for visualization)
                        for i in range(4):
                            token =self.sample_token(c)
                            img = self.sample_image(token)
                            axes[idx//interval, i].imshow(img, cmap='Greys')
                            axes[idx//interval, i].tick_params(
                                which='both',
                                bottom=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False
                            )
                        axes[idx//interval, 0].set_ylabel('%i' % idx)
                # zero optimizer gradients
                optimizer.zero_grad()
                # compute log-likelihood of the token
                score = self.score_type(c)
                score_list.append(score.item())
                # gradient descent step (minimize loss)
                loss = -score
                loss.backward()
                optimizer.step()
                # project all parameters into allowable range
                with torch.no_grad():
                    for param, lb, ub in zip(params, lbs, ubs):
                        if lb is not None:
                            torch.max(param, lb, out=param)
                        if ub is not None:
                            torch.min(param, ub, out=param)

            return score_list

        
        
    
    



## FUNCTIONS TO GENERATE NEW DATA #######################################################################################################





    def display_type(self,c):
        print('----BEGIN CHARACTER TYPE INFO----')
        print('num strokes: %i' % c.k)
        for i in range(c.k):
            print('Stroke #%i:' % i)
            print('\tsub-stroke ids: ', list(c.part_types[i].ids.numpy()))
            print('\trelation category: %s' % c.relation_types[i].category)
        print('----END CHARACTER TYPE INFO----')

    def test_dataset (self,n_strokes, nsub, alpha, perturbed):#,nb_iter):
        #lr=1e-3
        #eps=1e-4
        print('generating character...')
        lib = Library(use_hist=True)
        model = Perturbing()
        model.n_strokes = n_strokes
        subp_list = [10,20]
        model.subparts = subp_list #possible ids to use when sampling the id list 
        model.alpha = alpha 
        model.nsub = torch.tensor ([nsub])
       
        
        model.get_part_type(perturbed)
        c_type = model.known_stype(perturbed)
        model.display_type(c_type)
        print('THESE ARE THE IDS', model.ids)
        #print('num strokes: %i' % model.n_strokes)
        #print('num sub-strokes: ', model.nsub.item())
        
        
        
        # optimize the character type that we sampled
        #score_list = model.optimize_type(c_type, lr, nb_iter, eps)
        # plot log-likelihood vs. iteration
        #plt.figure()
        #plt.plot(score_list)
        #plt.ylabel('log-likelihood')
        #plt.xlabel('iteration')
        #plt.show()
    
    
    
   
    
        c_token = model.CM.sample_token(c_type)
        c_image = model.CM.sample_image(c_token)
        plt.rcParams["figure.figsize"] = [105, 105]
        #plt.imshow(c_image, cmap='Greys')
        plt.imsave('original.png', c_image, cmap='Greys')
            
            
        
    
    def get_primitive_board (self):
    
        
        #set the parameters to sample the primitives 
        self.n_strokes = 1 #we begin 1 stroke and 1 subtroke in order to visualize the primitives that are being used 
        self.nsub = torch.tensor ([1])
        subpart_idx = np.linspace(1,10,10, dtype = int)
        subpart_idx = subpart_idx.tolist()
        fig = plt.figure(figsize=(105, 105))
        rows = 2
        columns = 5
        
        
        for i in subpart_idx:
            
            subpart_sample = subpart_idx[i-1]
            
           
            self.subparts = subpart_sample
            self.sample_ids(False) 
            self.get_part_type(False)
            c_type = self.known_stype(False)
            c_token = self.CM.sample_token(c_type)
            c_image = self.CM.sample_image(c_token)
            fig.add_subplot(rows, columns, i)
           
           
            plt.axis('off')
            plt.title("prim_" + str(i))
            plt.imshow(c_image, cmap="Greys")
            #plt.show()
            
            
            
           
            
            
            
            #plt.subplot.set_title("prim"+str(i))
        plt.savefig('/Users/carolinacaramelo/Desktop/Thesis_results/primitives_all.png', cmap='Greys')
       
           
    
   
    
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
        

###### TESTING RECOVERY ######################################

#Exemplo para imagem que tem sempre apenas 1 sub-stroke para cada stroke
###LOAD AND TRASNFORM THE CELL ARRAYS FROM INFERENCE IN MATLAB TO NEEDED FORMS 

    def load(self):
        #IDS_TYPE
        ids = io.loadmat('./cell_ids.mat')

        items = ids.items()
        l =list(items)

        list_ids =l[len(l)-1]

        array_ids = np.array(list_ids)
        array_ids = (array_ids[1]).astype("float32")
        self.n_strokes=array_ids.size

        print(array_ids)


        

        ids=[]

        for i in range(array_ids.size):
            for j in range (array_ids[i].size):
                list_ids=[]
                
                array_idx = array_ids[i][j]
                
                print(array_ids)
                list_ids += [array_idx]
                
                ids += [torch.tensor(list_ids).long()]#in tensor form inside the list
                
        print("ids_1", ids)
        self.ids = ids
        
        #INVSCALE_TYPE
        
        invscale_type = io.loadmat("./invscale_type.mat")
        items = invscale_type.items()
        l =list(items)
        list_inv_type =l[len(l)-1]
        array_inv_type = np.array(list_inv_type)
        array_inv_type = (array_inv_type[1])
        
        list_inv_type=[]
        inv =[]
        for i in range(array_inv_type.size):
            list_inv_type += [(((array_inv_type[i])[0])[0])[0]]
            inv += [torch.tensor(list_inv_type)]#in tensor form 
        self.inv_type = inv
        print("invscale_1", inv)
        
            
        #NSUB TYPE
        nsub = io.loadmat('./nsub.mat')
        
        items = nsub.items()
        l =list(items)
        
        nsub =l[len(l)-1]
        
        array_nsub = np.array(nsub)
        array_nsub = (array_nsub[1])
        
        list_nsub=[]
        nsub = []
        
        for i in range(array_nsub.size):
            list_nsub += [(((array_nsub[i])[0])[0])[0]]
            nsub += [torch.tensor(list_nsub)]
        self.nsub = nsub
        print("nsub_1",nsub)
        
        #RELATION TYPE 
         
        #category   
        r = io.loadmat('./r_type.mat')
        
        items = r.items()
        l =list(items)
        
        list_r =l[len(l)-1]
        
        array_r = np.array(list_r)
        array_r = (array_r[1])
        
        list_r=[]
        
        for i in range(array_r.size):
            list_r += [((array_r[i])[0])[0]] #in list form 
        self.r = list_r
        print("relation category_1" , list_r)
        
        #gpos
        gpos = io.loadmat('./r_gpos.mat')
        
        items = gpos.items()
        l =list(items)
        
        gpos =l[len(l)-1]
        
        array_gpos = np.array(gpos)
        array_gpos = (array_gpos[1])
        
        gpos=[]
        list_gpos = []
        
        for i in range(array_gpos.size):
            print(i)
            try:
                a = (((array_gpos[i])[0])[0]).tolist()
                list_gpos += [torch.tensor([a])]
            
            except:
                list_gpos += [np.array([0,0]).tolist()]
                
                
            
        gpos += list_gpos#in tensor (1,2) form inside the list
            
        self.gpos = gpos
        print("gpos_1", gpos)
        
        #nprev
        nprev = io.loadmat('./r_nprev.mat')
        
        items = nprev.items()
        l =list(items)
        
        nprev =l[len(l)-1]
        
        array_nprev= np.array(nprev)
        array_nprev = (array_nprev[1])
        
        nprev=[]
        
        for i in range(array_nprev.size):
            nprev += [(((array_nprev[i])[0])[0])[0]] #in list form 
        self.nprev = nprev
        print("nprev_1", nprev)
        
        
        #ncpt
        ncpt = io.loadmat('./r_ncpt.mat')
        items = ncpt.items()
        l =list(items)
        
        ncpt =l[len(l)-1]
        
        array_ncpt= np.array(ncpt)
        array_ncpt = (array_ncpt[1])
        
        list_ncpt=[]
        ncpt=[]
        for i in range(array_ncpt.size):
            try:
                list_ncpt += [(((array_ncpt[i])[0])[0])[0]]
                
            except IndexError:
                list_ncpt += [0]
        ncpt += list_ncpt
            
        print("ncpt_1", ncpt) 
        self.ncpt_r = ncpt
        
        
        #subid_spot
        subid_spot = io.loadmat('./r_subid.mat')
        items = subid_spot.items()
        l =list(items)
        
        subid_spot =l[len(l)-1]
        
        array_subid_spot= np.array(subid_spot)
        array_subid_spot = (array_subid_spot[1])
        
        subid_spot=[]
        for i in range(array_subid_spot.size):
            try:
                subid_spot += [(((array_subid_spot[i])[0])[0])[0]]
                
            except IndexError:
                subid_spot += [0]
            
            
        self.subid_spot = subid_spot
        print("subid_1",subid_spot)
        
        #attach_spot
        attach_spot = io.loadmat('./r_attach_spot.mat')
        items = attach_spot.items()
        l =list(items)
        
        attach_spot =l[len(l)-1]
        
        array_attach_spot= np.array(attach_spot)
        array_attach_spot = (array_attach_spot[1])
        
        attach_spot=[]
        for i in range(array_attach_spot.size):
            try:
                attach_spot += [(((array_attach_spot[i])[0])[0])[0]]
                
            except IndexError:
                attach_spot +=[0]
            
            
        self.attach_spot = attach_spot
        print("attach_spot_1",attach_spot)
        
        #pos_token
        pos_token = io.loadmat('./pos_token.mat')
        items = pos_token.items()
        l =list(items)
        
        pos_token =l[len(l)-1]
        
        array_pos_token= np.array(pos_token)
        array_pos_token = (array_pos_token[1])
        
        
        
        pos=[]
        for i in range(array_pos_token.size):
            
                pos_token=[]
                pos_token += (((array_pos_token[i])[0])[0]).tolist()
                pos+= [torch.tensor(pos_token)]
                
           
            
            
        self.pos_token = pos
        print("pos_token_1", pos)
        
        
        #invscales_token
        inv_token = io.loadmat('./invscale_token.mat')
        items = inv_token.items()
        l =list(items)
        
        inv_token  =l[len(l)-1]
        
        array_inv_token = np.array(inv_token)
        array_inv_token  = (array_inv_token[1])
        
        
        inv=[]
        for i in range(array_inv_token.size):
                inv_token =[]
                inv_token  += [(((array_inv_token[i])[0])[0])[0]]
                inv+=[torch.tensor(inv_token)]
           
            
            
        self.inv_token  = inv
        print("inv_token_1", inv)
        
        #shapes_token
        shapes_token = io.loadmat('./shapes_token.mat')
        items = shapes_token.items()
        l =list(items)
        
        shapes_token=l[len(l)-1]
        
        array_shapes_token = np.array(shapes_token)
        array_shapes_token = (array_shapes_token[1])
        
       
        shapes=[]
        for i in range(array_shapes_token.size):
            shapes_token =[]
            for j in range(int((array_shapes_token[i][0]).size/2)):
                l =[]
                for s in range(2):
                    list_coo=[]
                    for k in range(self.nsub[0]):
                        list_coo+= [(array_shapes_token[i][k][j][s])]
                    l += [list_coo]
                shapes_token +=[l]                          
                                         
            shapes += [torch.tensor(shapes_token)]
                
             
        self.shapes_token  = shapes
        print("shapes_token_1", shapes)
        
        
        
        #eval_spot_token
        eval_spot_token = io.loadmat('./eval_spot_token.mat')
        items = eval_spot_token.items()
        l =list(items)
        
        eval_spot_token=l[len(l)-1]
        
        array_eval_spot_token = np.array(eval_spot_token)
        array_eval_spot_token = (array_eval_spot_token[1])
        
        eval_spot_token =[]
        for i in range(array_eval_spot_token.size):
            try:
                eval_spot_token += [(((array_eval_spot_token[i])[0])[0])[0]]
            
            except IndexError:
                eval_spot_token+=[0]
            
            
        self.eval_spot_token = eval_spot_token
        print("eval_spot_token_1", eval_spot_token)
        
        #epsilon
        epsilon = io.loadmat('./epsilon.mat')
        items = epsilon.items()
        l =list(items)
        
        epsilon=l[len(l)-1]
        
        array_epsilon = np.array(epsilon)
        array_epsilon = (array_epsilon[1])
        
        epsilon =[]
        for i in range(array_epsilon.size):
            epsilon += [((array_epsilon[i])[0])]
            
        
        self.r_epsilon = epsilon[0]
        print("epsilon_1", epsilon)
        
        
        #blur_sigma
        blur_sigma = io.loadmat('./blur_sigma.mat')
        items = blur_sigma.items()
        l =list(items)
        
        blur_sigma=l[len(l)-1]
        
        array_blur_sigma = np.array(blur_sigma)
        array_blur_sigma = (array_blur_sigma[1])
        
        blur_sigma =[]
        for i in range(array_blur_sigma.size):
            blur_sigma += [array_blur_sigma[i][0]]
            
        
        self.r_blur_sigma = blur_sigma[0]
        print("blur_sigma_1", blur_sigma)
        
    def load2(self):
        #IDS_TYPE
        ids = io.loadmat('./cell_ids2.mat')

        items = ids.items()
        l =list(items)

        list_ids =l[len(l)-1]

        array_ids = np.array(list_ids)
        array_ids = (array_ids[1]).astype("float32")
        self.n_strokes=array_ids.size

        print(array_ids)


        

        ids=[]

        for i in range(array_ids.size):
            for j in range (array_ids[i].size):
                list_ids=[]
                
                array_idx = array_ids[i][j]
                
                print(array_ids)
                list_ids += [array_idx]
                
                ids += [torch.tensor(list_ids).long()]#in tensor form inside the list
                
        print("ids_2", ids)
        self.ids = ids
        
        #INVSCALE_TYPE
        
        invscale_type = io.loadmat("./invscale_type2.mat")
        items = invscale_type.items()
        l =list(items)
        list_inv_type =l[len(l)-1]
        array_inv_type = np.array(list_inv_type)
        array_inv_type = (array_inv_type[1])
        
        list_inv_type=[]
        inv =[]
        for i in range(array_inv_type.size):
            list_inv_type += [(((array_inv_type[i])[0])[0])[0]]
            inv += [torch.tensor(list_inv_type)]#in tensor form 
        self.inv_type = inv
        print("invscale_2", inv)
        
            
        #NSUB TYPE
        nsub = io.loadmat('./nsub2.mat')
        
        items = nsub.items()
        l =list(items)
        
        nsub =l[len(l)-1]
        
        array_nsub = np.array(nsub)
        array_nsub = (array_nsub[1])
        
        list_nsub=[]
        nsub = []
        
        for i in range(array_nsub.size):
            list_nsub += [(((array_nsub[i])[0])[0])[0]]
            nsub += [torch.tensor(list_nsub)]
        self.nsub = nsub
        print("nsub_2",nsub)
        
        #RELATION TYPE 
         
        #category   
        r = io.loadmat('./r_type2.mat')
        
        items = r.items()
        l =list(items)
        
        list_r =l[len(l)-1]
        
        array_r = np.array(list_r)
        array_r = (array_r[1])
        
        list_r=[]
        
        for i in range(array_r.size):
            list_r += [((array_r[i])[0])[0]] #in list form 
        self.r = list_r
        print("relation category_2" , list_r)
        
        #gpos
        gpos = io.loadmat('./r_gpos2.mat')
        
        items = gpos.items()
        l =list(items)
        
        gpos =l[len(l)-1]
        
        array_gpos = np.array(gpos)
        array_gpos = (array_gpos[1])
        
        gpos=[]
        list_gpos = []
        
        for i in range(array_gpos.size):
            print(i)
            try:
                a = (((array_gpos[i])[0])[0]).tolist()
                list_gpos += [torch.tensor([a])]
            
            except:
                list_gpos += [np.array([0,0]).tolist()]
                
                
            
        gpos += list_gpos#in tensor (1,2) form inside the list
            
        self.gpos = gpos
        print("gpos_2", gpos)
        
        #nprev
        nprev = io.loadmat('./r_nprev2.mat')
        
        items = nprev.items()
        l =list(items)
        
        nprev =l[len(l)-1]
        
        array_nprev= np.array(nprev)
        array_nprev = (array_nprev[1])
        
        nprev=[]
        
        for i in range(array_nprev.size):
            nprev += [(((array_nprev[i])[0])[0])[0]] #in list form 
        self.nprev = nprev
        print("nprev_2", nprev)
        
        
        #ncpt
        ncpt = io.loadmat('./r_ncpt2.mat')
        items = ncpt.items()
        l =list(items)
        
        ncpt =l[len(l)-1]
        
        array_ncpt= np.array(ncpt)
        array_ncpt = (array_ncpt[1])
        
        list_ncpt=[]
        ncpt=[]
        for i in range(array_ncpt.size):
            try:
                list_ncpt += [(((array_ncpt[i])[0])[0])[0]]
                
            except IndexError:
                list_ncpt += [0]
        ncpt += list_ncpt
            
        print("ncpt_2", ncpt) 
        self.ncpt_r = ncpt
        
        
        #subid_spot
        subid_spot = io.loadmat('./r_subid2.mat')
        items = subid_spot.items()
        l =list(items)
        
        subid_spot =l[len(l)-1]
        
        array_subid_spot= np.array(subid_spot)
        array_subid_spot = (array_subid_spot[1])
        
        subid_spot=[]
        for i in range(array_subid_spot.size):
            try:
                subid_spot += [(((array_subid_spot[i])[0])[0])[0]]
                
            except IndexError:
                subid_spot += [0]
            
            
        self.subid_spot = subid_spot
        print("subid_2",subid_spot)
        
        #attach_spot
        attach_spot = io.loadmat('./r_attach_spot2.mat')
        items = attach_spot.items()
        l =list(items)
        
        attach_spot =l[len(l)-1]
        
        array_attach_spot= np.array(attach_spot)
        array_attach_spot = (array_attach_spot[1])
        
        attach_spot=[]
        for i in range(array_attach_spot.size):
            try:
                attach_spot += [(((array_attach_spot[i])[0])[0])[0]]
                
            except IndexError:
                attach_spot +=[0]
            
            
        self.attach_spot = attach_spot
        print("attach_spot_2",attach_spot)
        
        #pos_token
        pos_token = io.loadmat('./pos_token2.mat')
        items = pos_token.items()
        l =list(items)
        
        pos_token =l[len(l)-1]
        
        array_pos_token= np.array(pos_token)
        array_pos_token = (array_pos_token[1])
        
        
        
        pos=[]
        for i in range(array_pos_token.size):
            
                pos_token=[]
                pos_token += (((array_pos_token[i])[0])[0]).tolist()
                pos+= [torch.tensor(pos_token)]
                
           
            
            
        self.pos_token = pos
        print("pos_token_2", pos)
        
        
        #invscales_token
        inv_token = io.loadmat('./invscale_token2.mat')
        items = inv_token.items()
        l =list(items)
        
        inv_token  =l[len(l)-1]
        
        array_inv_token = np.array(inv_token)
        array_inv_token  = (array_inv_token[1])
        
        
        inv=[]
        for i in range(array_inv_token.size):
                inv_token =[]
                inv_token  += [(((array_inv_token[i])[0])[0])[0]]
                inv+=[torch.tensor(inv_token)]
           
            
            
        self.inv_token  = inv
        print("inv_token_2", inv)
        
        #shapes_token
        shapes_token = io.loadmat('./shapes_token2.mat')
        items = shapes_token.items()
        l =list(items)
        
        shapes_token=l[len(l)-1]
        
        array_shapes_token = np.array(shapes_token)
        array_shapes_token = (array_shapes_token[1])
        
       
        shapes=[]
        for i in range(array_shapes_token.size):
            shapes_token =[]
            for j in range(int((array_shapes_token[i][0]).size/2)):
                l =[]
                for s in range(2):
                    list_coo=[]
                    for k in range(self.nsub[0]):
                        list_coo+= [(array_shapes_token[i][k][j][s])]
                    l += [list_coo]
                shapes_token +=[l]                          
                                         
            shapes += [torch.tensor(shapes_token)]
                
             
        self.shapes_token  = shapes
        print("shapes_token_2", shapes)
        
        
        
        #eval_spot_token
        eval_spot_token = io.loadmat('./eval_spot_token2.mat')
        items = eval_spot_token.items()
        l =list(items)
        
        eval_spot_token=l[len(l)-1]
        
        array_eval_spot_token = np.array(eval_spot_token)
        array_eval_spot_token = (array_eval_spot_token[1])
        
        eval_spot_token =[]
        for i in range(array_eval_spot_token.size):
            try:
                eval_spot_token += [(((array_eval_spot_token[i])[0])[0])[0]]
            
            except IndexError:
                eval_spot_token+=[0]
            
            
        self.eval_spot_token = eval_spot_token
        print("eval_spot_token_2", eval_spot_token)
        
        #epsilon
        epsilon = io.loadmat('./epsilon2.mat')
        items = epsilon.items()
        l =list(items)
        
        epsilon=l[len(l)-1]
        
        array_epsilon = np.array(epsilon)
        array_epsilon = (array_epsilon[1])
        
        epsilon =[]
        for i in range(array_epsilon.size):
            epsilon += [((array_epsilon[i])[0])]
            
        
        self.r_epsilon = epsilon[0]
        print("epsilon_2", epsilon)
        
        
        #blur_sigma
        blur_sigma = io.loadmat('./blur_sigma2.mat')
        items = blur_sigma.items()
        l =list(items)
        
        blur_sigma=l[len(l)-1]
        
        array_blur_sigma = np.array(blur_sigma)
        array_blur_sigma = (array_blur_sigma[1])
        
        blur_sigma =[]
        for i in range(array_blur_sigma.size):
            blur_sigma += [array_blur_sigma[i][0]]
            
        
        self.r_blur_sigma = blur_sigma[0]
        print("blur_sigma_2", blur_sigma)
        
        
        
       
        


   
        
        
        


#NEW RELATION TYPE SAMPLING 
#We already have the relation type list, so we don't need to get it throught the previous parts  

    def sample_relation_type_recover(self, category, gpos, nprev, ncpt, subid, attach):
        """
        Sample a relation type from the prior for the current stroke,
        conditioned on the previous strokes

        Parameters
        ----------
        prev_parts : list of StrokeType
            previous part types - don't receive this, but instead already the category'

        Returns
        -------
        r : RelationType
            relation type sample
        """
       

        # now sample the category-specific type-level parameters
        if category == 'unihist':
            
            gpos = gpos
            # convert (1,2) tensor to (2,) tensor
            gpos = torch.squeeze(gpos)
            r = RelationIndependent(
                category, gpos, self.CTD.rdist.Spatial.xlim, self.CTD.rdist.Spatial.ylim
            )
        elif category in ['start', 'end', 'mid']:
            # sample random stroke uniformly from previous strokes. this is the
            # stroke we will attach to
            
            attach_ix = attach
            if category == 'mid':
                # sample random sub-stroke uniformly from the selected stroke
               
                attach_subix = subid
                # sample random type-level spline coordinate
                _, lb, ub = bspline_gen_s(ncpt, 1)
                eval_spot = dist.Uniform(lb, ub).sample()
                r = RelationAttachAlong(
                    category, attach_ix, attach_subix, eval_spot, ncpt
                )
            else:
                r = RelationAttach(category, attach_ix)
        else:
            raise TypeError('invalid relation')

        return r

###ALTERED FUNCTION TO SAMPLE CHARACTER TYPE 

    #altered function of StrokeTypeDist class, this time there will be a lot of things that we won't sample because we got the values from the inference
    def get_part_type_recover (self, perturbed, nsub, ids, invscales):
        """
        Define a stroke type from a list of sub-stroke ids
        Parameters, nsub and invscales, known from the inference phase 
        ----------
        ids : tensor
            vector, list of sub-part/stroke indices
        Returns
        -------
        p : StrokeType
            part type sample (strokes are the same things as parts)
        """
        # get the number of sub-strokes
        nsub = nsub
        # sample the sequence of sub-stroke IDs
        ids= ids
    

        # sample control points for each sub-stroke in the sequence
        shapes = self.STD.sample_shapes_type(ids)
        # sample scales for each sub-stroke in the sequence
        invscales = invscales
        p = StrokeType(nsub, ids, shapes, invscales)
        
        return p 


    
    #altered function from ConceptTypeDist class, defines a concept type
    def known_sample_type_recover (self,perturbed):
    # initialize relation type lists
        self.R = []
        self.P = []
        # for each part, sample part parameters
        for i in range(self.n_strokes):
            # sample the relation type
            r = self.sample_relation_type_recover(self.r[i], self.gpos[i], self.nprev[i], self.ncpt_r[i], self.subid_spot[i],self.attach_spot[i])
            p= self.get_part_type_recover(perturbed, self.nsub[i], self.ids[i], self.inv_type[i])
            # append to the lists
            self.R.append(r)
            self.P.append (p)
        ctype = ConceptType(self.n_strokes, self.P, self.R)

        return ctype



    #altered function from CharacterTypeDist class, defines a character type
    def known_ctype_recover(self, perturbed):
        
        self.known_sample_type_recover(perturbed)
        print(self.R, self.P)
        CT = ConceptType(self.n_strokes, self.P, self.R)
       
        #c = super(CharacterTypeDist, self).sample_type(self.n_strokes)
        #ctype = CharacterType(c.k, c.part_types, c.relation_types)

        return CharacterType(self.n_strokes, CT.part_types, CT.relation_types)
    
    
    #aaltered function from CharacterModel class, sampling the character type of the image (model) that will be generated
    def known_stype_recover(self, perturbed):
         return self.known_ctype_recover(perturbed)
    
    





#####FUNCTIONS FOR SAMPLING THE CHARACTER TOKEN  with known parameters without sampling     
    
    #Concept token
    def sample_token_recover(self, ctype):
        """
        Parameters
        ----------
        ctype : ConceptType

        Returns
        -------
        ctoken : ConceptToken
        """
        assert isinstance(ctype, ConceptType)
        self.P = []
        self.R = []
        for p, r in zip(ctype.part_types, ctype.relation_types):
            for i in range(self.n_strokes):
                # sample part token
                ptoken = self.sample_part_token_recover(p, self.shapes_token[i],self.inv_token[i])
                # sample relation token
                rtoken = self.sample_relation_token_recover(r, self.eval_spot_token[i])
                # sample part position from relation token
                ptoken.position = self.pos_token[i]
                # append them to the list
                self.P.append(ptoken)
                self.R.append(rtoken)
                
        ctoken = ConceptToken(self.P, self.R)
       

        return ctoken
    
    
    #Stroke token
    def sample_part_token_recover(self, ptype, shapes_token, inv_token):
        """
        Sample a stroke token

        Parameters
        ----------
        ptype : StrokeType
            stroke type to condition on

        Returns
        -------
        ptoken : StrokeToken
            stroke token sample
        """
        shapes_token = shapes_token
        invscales_token = inv_token
        ptoken = StrokeToken(
            shapes_token, invscales_token, self.STT.xlim, self.STT.ylim
        )

        return ptoken
    
    #character token 
    def samplech_token_recover(self, ctype):
        """
        Sample a character token from P(Token | Type = type).
        Note: should only be called from Model

        Parameters
        ----------
        ctype : CharacterType
            character type

        Returns
        -------
        ctoken : CharacterToken
            character token
        """
        # sample part and relation tokens
        #concept_token = super(CharacterTokenDist, self).sample_token_recover(ctype)
        self.sample_token_recover(ctype)
        CT = ConceptToken(self.P, self.R)
        

        # sample affine warp
        #A = self.sample_affine()
        A = None

        # sample image noise
        #epsilon = self.sample_image_noise()
        epsilon = self.r_epsilon

        # sample image blur
        #blur_sigma = self.sample_image_blur()
        blur_sigma = self.r_blur_sigma

        # create the character token
        ctoken = CharacterToken(
            CT.part_tokens, CT.relation_tokens, A,
            epsilon, blur_sigma
        )

        return ctoken
    
    #character model 
    def model_sample_token_recover(self, ctype):
        return self.samplech_token_recover(ctype)
    
    
    
    #relation token
    def sample_relation_token_recover(self, rtype, eval_spot):
        """
        Parameters
        ----------
        rtype : Relation

        Returns
        -------
        rtoken : RelationToken
        """
        if rtype.category == 'mid':
            assert hasattr(rtype, 'eval_spot')
            
            eval_spot_token = eval_spot
            rtoken = RelationToken(rtype, eval_spot_token=eval_spot_token)
        else:
            rtoken = RelationToken(rtype)

        return rtoken




 





        
