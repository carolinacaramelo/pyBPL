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
from statistics import *
from DP import DirichletProcessSample
import torch
import torch.distributions as dist
import os 
from PIL import Image as img
from pybpl.model.type_dist import ConceptTypeDist, CharacterTypeDist, PartTypeDist, StrokeTypeDist, RelationTypeDist
from pybpl.model.token_dist import ConceptTokenDist, CharacterTokenDist, PartTokenDist, StrokeTokenDist, RelationTokenDist
from scipy import io
from scipy.stats import norm


class Perturbing (StrokeTypeDist, ConceptTypeDist, CharacterModel, Library):
    __relation_categories = ['unihist', 'start', 'end', 'mid']
    def __init__ (self):
        # Generative process relevant variables 
        #dcan be previously defined 
        #self.ids = torch.tensor([1]) # ids of the substrokes
        self.n_strokes = torch.tensor(1) # number of strokes, it has to be <=10
        self.subparts = [30,300,45,70,89] #possible ids to use when sampling the id list 
        self.nsub_list = [torch.tensor (1)]
        self.nsub = self.nsub_list[0]#number of subparts its also <=10, in the normal case it depends on n_strokes
        self.R = []
        self.P = []
        self.seq_ids = [[]]
        
       
       
       


        # initializaton of class inheritances
        self.lib = Library(use_hist = True)
        self.STD = StrokeTypeDist(self.lib)
        self.CTD = ConceptTypeDist(self.lib)
        self.CHTD = CharacterTypeDist(self.lib)
        self.CM = CharacterModel(self.lib)
        self.RTD = RelationTypeDist(self.lib)
        self.STT = StrokeTokenDist(self.lib)
        self.CTT = ConceptTokenDist(self.lib)
        self.RTT = RelationTokenDist(self.lib)
        self.CT = CharacterType(self.n_strokes, self.P, self.R)
        #self.DP = DirichletProcessSample(base_measure=self.base_measure,alpha=self.alpha)
        
  
        
#### FUNCTIONS FOR SAMPLING THE CHARACTER MODEL TYPE ################################################################################################
    
    #altered function of StrokeTypeDist class, defines a stroke type 
    def get_part_type (self, perturbed, *args):
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
        self.sample_ids(perturbed, *args)

        # sample control points for each sub-stroke in the sequence
        shapes = self.STD.sample_shapes_type(self.ids)
        self.shapes = shapes
        # sample scales for each sub-stroke in the sequence
        invscales = self.STD.sample_invscales_type(self.ids)
        self.invscales = invscales
        print(self.invscales)
        print(self.shapes)
        p = StrokeType(nsub, self.ids, self.shapes, self.invscales)
        
        return p 
       
    
    #altered function from ConceptTypeDist class, defines a concept type
    def known_sample_type (self,perturbed, *args):
    # initialize relation type lists
        self.R = []
        self.P = []
        # for each part, sample part parameters
        
        for i in range(self.n_strokes):
            self.nsub = self.nsub_list[i]
            # sample the relation type
            p= self.get_part_type(perturbed, *args)
            r = self.CTD.rdist.sample_relation_type(self.P)
            # append to the lists
            self.R.append(r)
            self.P.append (p)
        ctype = ConceptType(self.n_strokes, self.P, self.R)

        return ctype
 

    #altered function from CharacterTypeDist class, defines a character type
    def known_ctype (self, perturbed, *args):
        
        self.known_sample_type(perturbed, *args)
        print(self.R, self.P)
        CT = ConceptType(self.n_strokes, self.P, self.R)

        return CharacterType(self.n_strokes, CT.part_types, CT.relation_types)
    
    
    #altered function from CharacterModel class, sampling the character type of the image (model) that will be generated
    def known_stype(self, perturbed, *args):
         return self.known_ctype(perturbed, *args)

   
#### PERTURBED SAMPLING - KNOWING PRIMITIVES USED #############################################################################################################
    
    #subparts are the available ids, nsub are the number of ids we want to sample (the number of subparts in the id list sequence)
    def sample_ids (self, perturbed, *args): #perturbed : bool, alpha
        # sub-stroke sequence is a list, this is the list of ids that we will sample for one stroke, list of ids of substrokes for one stroke
        ids =[]
        if perturbed:
            if args == "perturb_quartile1":
                #set the initial transition probabilities 
                pT = perturb_quartile1()[1] #perturbed logstart
            if args == "perturb_quartile2":
                pT = perturb_quartile2()[1] #perturbed logstart
            if args == "perturb_q4":
                pT = perturb_q4()[1]
            if args == "perturb_flatenning":
                pT = perturb_flatenning()[1]
            if args == "perturb_zero":
                pT = perturb_zero()[1]
            if args == "perturb_specific":
                pT = perturb_specific()[1]
            if args == "perturb_specific2":
                pT = perturb_specific2()[1]
      
            pT = self.truncate_start(pT)
            #step through and sample 'nsub' sub-strokes
            for _ in range(self.nsub):
                #ss = torch.multinomial(pT,1)
                #ss= torch.squeeze (ss,-1)
                ss = dist.Categorical(probs=pT).sample()
                ids.append(ss)
                # update transition probabilities; condition on previous sub-stroke
                pT = self.pT_perturbed(ss, *args)
        else: 
            #set the initial transition probabilities 
            pT = torch.exp(self.lib.logStart)
            pT = self.truncate_start(pT)
            for _ in range(self.nsub):
                #ss = torch.multinomial(pT,1)
                #ss= torch.squeeze (ss,-1)
                ss = dist.Categorical(probs=pT).sample()
                ids.append(ss)
                print (ids)
                # update transition probabilities; condition on previous sub-stroke
                pT = self.pT_truncate(ss)
                
        
        # convert list into tensor
        self.pT = pT #isto é irrelevante estar aqui, pois para cada stroke recomeça-se o sampling dos ids com a logStart
        self.seq = ids
        self.ids = torch.stack(ids)
        return ids

    #remove entries that don't correspond to the available indexes (subids) of primitives given by the subpart list in the logStart matrix 
    def truncate_start (self, pT):
        m = np.isin([i for i in range(len(pT))], self.subparts)
        m = np.invert(m)
        indices = np.argwhere(m)
        pT[indices] = 0
        "what is commented is needed to get the primitive board"
        #if pT[self.subparts] == 0: 
              #pT[self.subparts]= 1
        assert pT.sum() != 0, "pT vector sum is zero"
        return pT
    
    #for the non perturbed pT, only account for the primitives that are comprised in the subpart list, other primitives other than those cannot be sampled
    def pT_truncate (self,ids):
        assert ids.shape == torch.Size([])
        R = self.truncateT()
        R = R[ids]
        #print(R)
        pT = R / torch.sum(R) #normalize pT
        return pT
    
   
   #remove entries that don't correspond to the possible indexes
    def truncateT (self):
        logR = self.lib.logT
        R = torch.exp(logR)
        m = np.isin([i for i in range(len(R))], self.subparts)
        n = np.invert(m)
        indices2 = np.argwhere(n)
        R[:,indices2]=0
        R[indices2,:]=0
       
        return R
    
    #def M_perturbation(self,str, alpha): #vamos ter de mudar esta função para corresponder às funções desenvolvidas em statistics - podemos apagar isto
        #logR = self.lib.logT
        #s = list(logR.size())
        #if str == "Gaussian":
            #M_perturbation = (alpha**0.5) * torch.randn(s[0], s[1]) #The function torch.randn produces a tensor with elements drawn from a Gaussian distribution of zero mean and unit variance. Multiply by sqrt(0.1) to have the desired variance.
        #if str == "Dirichlet":
        #if str == "Poisson":
        #if str == "Exponential":
        #return M_perturbation
            

    def truncateT_perturbed (self, *args): #in gaussian noise the alpha can be sqrt(variance) #perturbar antes a matrix logT, perturbar sem normalizar na logT 
        if args == "perturb_quartile1":
            #set the initial transition probabilities 
            pT = statistics.perturb_quartile1()[0] #perturbed logstart
        if args == "perturb_quartile2":
            pT = statistics.perturb_quartile2()[0] #perturbed logstart
        if args == "perturb_q4":
            pT = statistics.perturb_q4()[0]
        if args == "perturb_flatenning":
            pT = statistics.perturb_flatenning()[0]
        if args == "perturb_zero":
            pT = statistics.perturb_zero()[0]
        if args == "perturb_specific":
            pT = statistics.perturb_specific()[0]
        if args == "perturb_specific2":
            pT = statistics.perturb_specific2()[0]
            
        m = np.isin([i for i in range(len(pT))], self.subparts)
        n = np.invert(m)
        indices2 = np.argwhere(n)
        pT[:,indices2]=0
        pT[indices2,:]=0
        
        return pT
   
    def pT_perturbed (self,ids, *args): 
        assert ids.shape == torch.size ([])
        pT = self.truncateT_perturbed(*args)
        pT = pT[ids]
        pT_perturbed = pT
        return pT_perturbed
        
     
#o objetivo é dar uma lista de ids possíveis a utilizar e a partir daí fazer o sampling da ordem em que os ids aparecem usando pT 
# ou então a pT perturbada, por isso o que vamos ter é uma função que recebe uma lista de ids e usa pT, outra função que recebe a 
# a lista de ids e usa a pT perturbed, e uma função que recebe como argumento a lista restrita de ids e diz se perturbamos ou não 
# o que esta última função vai fazer é cuspir uma lista desses ids ordenados de maneira diferente - depois corremos o exemplo com
# essa lista de ids e vemos a diferença entre a imagem gerada com pT ou com pT perturbed 
       

###### SCORING SAMPLING #########################################################################################################################

#we can also get the scores for the perturbed version of the model 
# we are not sampling n_strokes and nsub from the distributions but we are scoring the values we are giving these variables from the correspondent distributions 


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
        if self.n_strokes > len(self.CHTD.kappa.probs):
            ll = torch.tensor(-float('Inf'))
        else:
            # score points using kappa
            # NOTE: subtract 1 to get 0-indexed samples
            ll = self.CHTD.kappa.log_prob(self.n_strokes-1)
    
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
        pvec = self.STD.pmat_nsub[self.n_strokes-1]
        # make sure pvec is a vector
        assert len(pvec.shape) == 1
        # score using the categorical distribution
        ll = dist.Categorical(probs=pvec).log_prob(self.nsub-1)

        return ll
    
    def score_subIDs(self): #isto ainda não está a contabilizar as perturbações
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
        pT = torch.exp(self.lib.logStart)
        
        # initialize log-prob vector
        ll = torch.zeros(len(self.ids), dtype=torch.float)
        # step through sub-stroke IDs
        for i, ss in enumerate(self.ids):
            # add to log-prob accumulator
            ll[i] = dist.Categorical(probs=pT).log_prob(ss)
            # update transition probabilities; condition on previous sub-stroke
            pT = self.pT_truncate(ss)

        return ll
    
    
    def score_shapes_type(self): #stays the same 
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
        #if self.isunif:
            #raise NotImplementedError
        # check that subid is a vector
        assert len(self.ids.shape) == 1
        # record vector length
        nsub = len(self.ids)
        assert self.shapes.shape[-1] == nsub
        # reshape tensor (ncpt, 2, nsub) -> (ncpt*2, nsub)
        shapes = self.shapes.view(self.ncpt*2,nsub)
        # transpose axes (ncpt*2, nsub) -> (nsub, ncpt*2)
        shapes =shapes.transpose(0,1)
        # create multivariate normal distribution
        mvn = dist.MultivariateNormal(
            self.STD.shapes_mu[self.ids], self.STD.shapes_Cov[self.ids]
        )
        # score points using the multivariate normal distribution
        ll = mvn.log_prob(shapes)

        return ll
    
    def score_invscales_type(self): #also stays the same, we are not changing the way we sample shapes and scale 
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
        assert len(self.invscales.shape) == 1
        assert len(self.ids.shape) == 1
        assert len(self.invscales) == len(self.ids)
        # create gamma distribution
        gamma = dist.Gamma(self.STD.scales_con[self.ids], self.STD.scales_rate[self.ids])
        # score points using the gamma distribution
        ll = gamma.log_prob(self.invscales)

        return ll
    
    def score_part_type(self):#ptype is gotten from the get_part_type function, it is the stroke type 
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
        
        nsub_score = self.score_nsub()
        subIDs_scores = self.score_subIDs()
        shapes_scores = self.STD.score_shapes_type(self.ids, self.shapes)
        invscales_scores = self.STD.score_invscales_type(self.ids, self.invscales)
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
        ll = ll + self.score_n_strokes()
        # step through and score each part
        for i in range(self.n_strokes):
            ll = ll + self.score_part_type()
            ll = ll + self.CTD.rdist.score_relation_type(
                self.P[:i], self.R[i]
            )

        return ll
    

######## OPTIMIZING CHARACTER MODEL TYPE ##################################
    #adapted optimize character type to perturbing 
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
                        token =self.CM.sample_token(c)
                        img = self.CM.sample_image(token)
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
                    if torch.is_tensor(lb):
                        torch.max(param, lb, out=param)
                    if torch.is_tensor(ub):
                        torch.min(param, ub, out=param)

        return score_list
    
    def optimize(self,c, lr, nb_iter, eps):
        # optimize the character type that we sampled
        score_list = self.optimize_type(c, lr, nb_iter, eps)
        # plot log-likelihood vs. iteration
        plt.figure(figsize=(10,10))
        plt.plot(score_list)
        plt.ylabel('log-likelihood')
        plt.xlabel('iteration')
        plt.show()
        
    def optimize_test(self):
        for i in range(10):
            c_type = self.known_stype(False)
            self.optimize(c_type, lr=1e-3,nb_iter= 1000,eps=1e-4 )
       
        


## FUNCTIONS TO GENERATE NEW DATA/ NEW CHARACTERS  #######################################################################################################


    def display_type(self,c):
        print('----BEGIN CHARACTER TYPE INFO----')
        print('num strokes: %i' % c.k)
        for i in range(c.k):
            print('Stroke #%i:' % i)
            print('\tsub-stroke ids: ', list(c.part_types[i].ids.numpy()))
            print('\trelation category: %s' % c.relation_types[i].category)
        print('----END CHARACTER TYPE INFO----')
        
        file = open("./original_info", 'w') 
        file.write("num strokes:" + str(c.k)+ "\n")
        for i in range(c.k):
            file.write("Stroke #:" + str(i)+ "\n")
            file.write("\tsub-stroke ids:" + str(list(c.part_types[i].ids.numpy()))+ "\n")
            file.write("\\trelation category:" + str(c.relation_types[i].category)+ "\n")
            file.write("----END CHARACTER TYPE INFO----")
        file.close()

    def test_dataset (self,n_strokes, nsub, perturbed):
        """
        This function allows us to generate a new character type
        generates a character type given previously the number of strokes, the number of substrokes for each stroke and the indexes of the primitives 
        that we want to use (main difference from the original code - we control n_strokes, nsub and subparts)
        after generating a character type ot optimizes the type - minimizing the loglikelihood score of the character type 

        Parameters
        ----------
        n_strokes: int, number of strokes
        nsub : list of tensors dim= n_strokes, each tensor has the nsub for each stroke
        nb_iter : int
        perturbed = bool

        """

        print('generating character...')
        model = Perturbing()
        model.n_strokes = n_strokes #number of strokes 
        subp_list =[655,962,908,76,297, 100, 15, 20, 1000, 1289, 450, 500, 1212,75,53,62] #list of primitive indexes to use 
        model.subparts = subp_list #possible ids to use when sampling the id list 
        model.nsub_list = nsub #list of nsub
        
       
        c_type = model.known_stype(perturbed)
        model.display_type(c_type)
        c_token = model.CM.sample_token(c_type)
        c_image = model.CM.sample_image(c_token)
        plt.rcParams["figure.figsize"] = [105, 105]
        plt.imshow(c_image, cmap='Greys')
        plt.imsave('./original.png', c_image, cmap='Greys')

    def test_dataset2(self,n_strokes, nsub, perturbed ,nb_iter):
        """
        This function allows us to generate a new character type
        generates a character type given previously the number of strokes, the number of substrokes for each stroke and the indexes of the primitives 
        that we want to use (main difference from the original code - we control n_strokes, nsub and subparts)
        after generating a character type ot optimizes the type - minimizing the loglikelihood score of the character type 

        Parameters
        ----------
        n_strokes: int, number of strokes
        nsub : list of tensors dim= n_strokes, each tensor has the nsub for each stroke
        nb_iter : int
        perturbed = bool

        """
        
        lr=1e-3 #default learning rate 
        eps=1e-4 #default tolerance for constrained optimization
        
        print('generating character...')
        model = Perturbing()
        model.n_strokes = n_strokes #number of strokes 
        subp_list =[655,962,908,76,297] #list of primitive indexes to use 
        model.subparts = subp_list #possible ids to use when sampling the id list 
        model.nsub_list = nsub #list of nsub
        
       
        c_type = model.known_stype(perturbed)
        model.display_type(c_type)

        # optimize the character type that we sampled
        model.optimize(c_type,lr, nb_iter, eps)

        c_token = model.CM.sample_token(c_type)
        c_image = model.CM.sample_image(c_token)
        plt.rcParams["figure.figsize"] = [105, 105]
        plt.imshow(c_image, cmap='Greys')
        plt.imsave('./original.png', c_image, cmap='Greys')
            
            
        
    #visualize every primitive 
    def get_primitive_board (self):
        #set the parameters to sample the primitives 
        self.n_strokes = 1 #we begin 1 stroke and 1 subtroke in order to visualize the primitives that are being used 
        self.nsub = torch.tensor (1) #sampling every existing primitive, 1 stroke and 1 substroke
        #list of subparts that will allows us to sample each primitive at a time 
        subpart_idx = np.linspace(0,1211,1212, dtype = int)
        subpart_idx = subpart_idx.tolist()
    
        rows = 34
        columns = 36
        fig ,ax= plt.subplots(nrows=rows,ncols=columns,figsize=(105,105))

        count_images=0
   
        for i in range(rows):
            for j in range(columns):
                try:
                    #for each character that is being generated the subparts available will only be 1 primitive, do this for every existing primitive
                    subpart_sample = subpart_idx[count_images]
                    self.subparts = subpart_sample
                    
                    c_type = self.known_stype(False)
                    print("done", count_images)
                    print(self.ids)
                    print(self.pT)
                    
                    c_token = self.CM.sample_token(c_type)
                    c_image = self.CM.sample_image(c_token)
                   
                    ax[i,j].imshow(c_image, cmap="Greys")
                    ax[i,j].set_title("prim_" + str(count_images),fontsize=20)
                    count_images+=1
                except:
                    pass           
        fig.tight_layout()
        #plt.show()
        fig.savefig('/Users/carolinacaramelo/Desktop/Thesis_results/primitives_all2.png', cmap='Greys')
       
           
    
   
    
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
        array_ids = (array_ids[1])
        
        self.n_strokes=array_ids.size

        print(array_ids)

        ids=[]

        for i in range(array_ids.size):
            list_ids=[]
            for j in range (array_ids[i][0].size):
                
                #for k in range (array_ids[0][0][i].size):
                    #list_ids=[]
                
                array_idx = array_ids[i][0][j][0].astype("float32")
                
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
        
       
        nsub = []
        
        for i in range(array_nsub.size):
            list_nsub=[]
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
       
        l =[]
        
        for i in range(array_gpos.size):
            list_gpos = []
            for s in range(2):
                
                try:
                    a1 = (((array_gpos[i])[0])[0])[s]
                    a2 = [a1]
                    list_gpos += a2
                
                except IndexError:
                    b = [0]
                    print(b)
                    list_gpos += b
            l += [torch.tensor(list_gpos)]
                    
                
            
        gpos += l#in tensor (1,2) form inside the list
            
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
        for i in range(self.n_strokes):
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
        for i in range(self.n_strokes):
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
        for i in range(self.n_strokes):
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
            for j in range (self.nsub[i].item()):
          
                    inv_token  += [(((array_inv_token[i])[0])[j])[0]]
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
            if array_shapes_token[i][0].size == 10:
                for j in range(int((array_shapes_token[i][0]).size/2)):
                    l =[]
                    for s in range(2):
                        list_coo=[]
                        list_coo+= [(array_shapes_token[i][0][j][s])]
                        l += [list_coo]
      
                    shapes_token +=[l]
                shapes += [torch.tensor(shapes_token)]
        
            else:
                for j in range(5):
                    l =[]
                    for k in range(2):
                        list_coo=[]
                        #for s in range(self.nsub[i].item()):
                        list_coo+= (array_shapes_token[i][0][j][k].tolist())
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

###ALTERED FUNCTION TO SAMPLE CHARACTER TYPE - known parameters from inference - without sampling 

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


####### GENERATE ALPHABET #######################################################################################

 
    def p_mem(self, perturbed, *args): #perturbed, *args - fazer no final para pT 
       #this function will transform the learnt probability distributions of the BPL parameters in a new stochastic memoized distribution
       #P-mem, through a Dirichlet Process, in order to create dependencies between the different characters of a new generated alphabet. 
       #P-mem will induce dependencies between previously independent samples. This new "memoized" procedures will define a set of probability distributions 
       #with the chinese restaurant process clustering property, used for learning the number of strokes, sub-strokes, the strokes (subids sequence, shapes, invscales)
       # and the relation types that are characteristic of a particular alphabet. 
       # The function p_mem will return a collection of these procedures that will be passed to generate_type functions. 
       
       #P-mem list
       Pmem=[]
       
       #alpha is defined the same for every DP (is it?)
       alpha = 3
       
       #P-mem for number of strokes 
       base_measure = lambda: dist.Categorical(probs=self.lib.pkappa[0:7]).sample()+1  #distribution for sampling n_strokes #only allowing 7 strokes 
       dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
       Pmem +=[[dirichlet_norm]]
       
       
       #P-mem for number of substrokes 
       Pmem_nsub = []
       for i in range(10):
           pvec = self.lib.pmat_nsub[i][0:8] #only allowing 8 nsub
           base_measure =  lambda: dist.Categorical(probs=pvec).sample() + 1 #distribution for sampling nsub, depending on the number of n_strokes
           dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
           Pmem_nsub += [dirichlet_norm]
       Pmem += [Pmem_nsub]
       
       #P-mem for GENERATE STROKES - includes subids, shapes and scales of strokes 
       #P-mem for subIDS - logStart
       if perturbed:
           if args == "perturb_quartile1":
               #set the initial transition probabilities 
               logStart = perturb_quartile1()[1] #perturbed logstart
           if args == "perturb_quartile2":
               logStart = perturb_quartile2()[1] #perturbed logstart
           if args == "perturb_q4":
               logStart = perturb_q4()[1]
           if args == "perturb_flatenning":
               logStart = perturb_flatenning()[1]
           if args == "perturb_zero":
               logStart = perturb_zero()[1]
           if args == "perturb_specific":
               logStart = perturb_specific()[1]
           if args == "perturb_specific2":
               logStart = perturb_specific2()[1]
       
       else: 
           logStart = torch.exp(self.lib.logStart)
       
       base_measure = lambda: dist.Categorical(probs=logStart).sample()  #distribution for sampling the first nsub of each strokes
       dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
       Pmem +=[[dirichlet_norm]]
       
       
       #P-mem for subIDS - pT - including perturbations
       Pmem_pT = []
       if perturbed:
           if args == "perturb_quartile1":
               #set the initial transition probabilities 
               pT = perturb_quartile1()[0] #perturbed logstart
           if args == "perturb_quartile2":
               pT = perturb_quartile2()[0] #perturbed logstart
           if args == "perturb_q4":
               pT = perturb_q4()[0]
           if args == "perturb_flatenning":
               pT = perturb_flatenning()[0]
           if args == "perturb_zero":
               pT = perturb_zero()[0]
           if args == "perturb_specific":
               pT = perturb_specific()[0]
           if args == "perturb_specific2":
               pT = perturb_specific2()[0]
           self.pT = pT
       else:
            logR = self.lib.logT
            R = torch.exp(logR)
            pT = R / torch.sum(R)
            self.pT = pT
       for i in range(1212):  
            pT = self.pT
            prev_state = torch.tensor([i])
            pT = pT[prev_state]
            base_measure =  lambda: dist.Categorical(probs=pT).sample() #distribution for sampling subids, depending on previous state
            dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
            Pmem_pT+= [dirichlet_norm]
       Pmem += [Pmem_pT]        
          

       #P-mem for relation type
       base_measure = lambda: dist.Categorical(probs=self.lib.rel['mixprob']).sample() #distribution for sampling n_strokes 
       dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
       Pmem +=[dirichlet_norm]
       
       self.Pmem = Pmem
       return Pmem
   
#how to acess Pmem list - saved list with "memoized" distributions - we will sample one Pmem list for each different alphabet we are originating 
# Pmem[0][0] - k
# Pmem[1][...] - nsub
# Pmem[2][0] - logstart
# Pmem[3][...] - pT
# Pmem[4] - relation type


    def p_mem2(self):
        Pmem=[]
        alpha = 0.1
        #P-mem for shapes
        shapes_mu = self.lib.shape['mu'][self.ids]
        shapes_Cov = self.lib.shape['Sigma'][self.ids]
        base_measure =  lambda: dist.MultivariateNormal(shapes_mu,shapes_Cov).sample() #distribution for sampling shapes, depending on previous subID
        dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
        Pmem += [dirichlet_norm]
        
        #P-mem for invscales
        scales_theta = self.lib.scale['theta']
        scales_con = scales_theta[:,0][self.ids] 
        scales_rate = 1 / scales_theta[:,1][self.ids]
        base_measure =  lambda: dist.Gamma(scales_con, scales_rate).sample() #distribution for sampling invscales, depending on previous 
        dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
        Pmem += [dirichlet_norm]
        return Pmem
    
# Pmem[0] - shapes
# Pmem[1]- invscales

############################ SAMPLING PARAMETERS WITH DIRICHLET PROCESS - TYPE LEVEL #################################

    def DP_sample_relation_type(self, prev_parts):
        """
        Sample a relation type from the prior for the current stroke,
        conditioned on the previous strokes

        Parameters
        ----------
        prev_parts : list of StrokeType
            previous part types

        Returns
        -------
        r : RelationType
            relation type sample
        """
        for p in prev_parts:
            assert isinstance(p, StrokeType)
        nprev = len(prev_parts)
        stroke_ix = nprev
        # first sample the relation category
        if nprev == 0:
            category = 'unihist'
        else:
            indx = self.Pmem[4]() #change is here - sampling from Pmem instead of sampling from original distribution 
            category = self.__relation_categories[indx]

        # now sample the category-specific type-level parameters
        if category == 'unihist':
            data_id = torch.tensor([stroke_ix])
            gpos = self.RTD.Spatial.sample(data_id)
            # convert (1,2) tensor to (2,) tensor
            gpos = torch.squeeze(gpos)
            r = RelationIndependent(
                category, gpos, self.RTD.Spatial.xlim, self.RTD.Spatial.ylim
            )
        elif category in ['start', 'end', 'mid']:
            # sample random stroke uniformly from previous strokes. this is the
            # stroke we will attach to
            probs = torch.ones(nprev)
            attach_ix = dist.Categorical(probs=probs).sample()
            if category == 'mid':
                # sample random sub-stroke uniformly from the selected stroke
                nsub = prev_parts[attach_ix].nsub
                probs = torch.ones(nsub)
                attach_subix = dist.Categorical(probs=probs).sample()
                # sample random type-level spline coordinate
                _, lb, ub = bspline_gen_s(self.RTD.ncpt, 1)
                eval_spot = dist.Uniform(lb, ub).sample()
                r = RelationAttachAlong(
                    category, attach_ix, attach_subix, eval_spot, self.RTD.ncpt
                )
            else:
                r = RelationAttach(category, attach_ix)
        else:
            raise TypeError('invalid relation')

        return r

    def DP_sample_k(self):
        """
        Sample a stroke count from the prior

        Returns
        -------
        k : tensor
            scalar; stroke count
        """
        # sample from kappa
        # NOTE: add 1 to 0-indexed samples
        k = self.Pmem[0][0]() #change is here, instead of sampling from original distribution we sample from Pmem
        self.n_strokes = k
        return k
    
    def DP_sample_nsub(self):
        """
        Sample a sub-stroke count

        Parameters
        ----------
        k : tensor
            scalar; stroke count

        Returns
        -------
        nsub : tensor
            scalar; sub-stroke count
        """
       
        # sample from the categorical distribution. Add 1 to 0-indexed sample
        nsub = self.Pmem[1][self.n_strokes-1]()
        self.nsub = nsub

        return nsub

        
    def DP_sample_subIDs(self):
        """
        Sample a sequence of sub-stroke IDs from the prior

        Parameters
        ----------
        nsub : tensor
            scalar; sub-stroke count

        Returns
        -------
        subid : (nsub,) tensor
            sub-stroke ID sequence
        """
        # nsub should be a scalar
        assert self.nsub.shape == torch.Size([])
        # set initial transition probabilities
        DP = self.Pmem[2][0]
       
        # sub-stroke sequence is a list
        subid = []
        # step through and sample 'nsub' sub-strokes
        for _ in range(self.nsub):
            # sample the sub-stroke
            ss = DP()
            print(ss)
            ss = ss.squeeze()
            subid.append(ss)
            print(ss)
            print(subid)
            # update transition probabilities; condition on previous sub-stroke
            DP = self.Pmem[3][ss]
        # convert list into tensor
        subid = torch.stack(subid)
        self.ids = subid

        return subid

    def DP_sample_shapes_type(self):
        """
        Sample the control points for each sub-stroke ID in a given sequence

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        shapes : (ncpt, 2, nsub) tensor
            sampled shapes of bsplines
        """
        if self.STD.isunif:
            raise NotImplementedError
        # check that subid is a vector
        assert len(self.ids.shape) == 1
        # record vector length
        nsub = len(self.ids)
        
        shapes = self.p_mem2()[0]()
       
        # transpose axes (nsub, ncpt*2) -> (ncpt*2, nsub)
        shapes = shapes.transpose(0,1)
        # reshape tensor (ncpt*2, nsub) -> (ncpt, 2, nsub)
        shapes = shapes.view(self.STD.ncpt,2,nsub)
        self.shapes= shapes

        return shapes

    def DP_sample_invscales_type(self):
        """
        Sample the scale parameters for each sub-stroke

        Parameters
        ----------
        subid : (nsub,) tensor
            sub-stroke ID sequence

        Returns
        -------
        invscales : (nsub,) tensor
            scale values for each sub-stroke
        """
        if self.STD.isunif:
            raise NotImplementedError
        # check that it is a vector
        assert len(self.ids.shape) == 1
        invscales = self.p_mem2()[1]()
        
       
        self.invscales= invscales  
        print(invscales)
        return invscales


##################################### SAMPLING CHARACTER TYPE WITH DP ##############################

    #altered function of StrokeTypeDist class, defines a stroke type 
    def DP_get_part_type (self, perturbed, *args):
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
        self.DP_sample_nsub()
        # sample the sequence of sub-stroke IDs
        self.DP_sample_subIDs()
        # sample control points for each sub-stroke in the sequence
        self.DP_sample_shapes_type()
        #sample scales for each sub-stroke in the sequence
        self.DP_sample_invscales_type()
        print(self.shapes)
        print(self.invscales)
        
        p = StrokeType(self.nsub, self.ids, self.shapes, self.invscales)
        
        return p 
       
    
    #altered function from ConceptTypeDist class, defines a concept type
    def DP_sample_type (self,perturbed, *args):
        self.DP_sample_k()
        #initialize relation type lists
        self.R = []
        self.P = []
       
        for i in range(self.n_strokes):
            
            # sample the relation type
            p= self.DP_get_part_type(perturbed, *args)
            r = self.DP_sample_relation_type(self.P)
            # append to the lists
            self.R.append(r)
            self.P.append (p)
        ctype = ConceptType(self.n_strokes, self.P, self.R)

        return ctype
 

    #altered function from CharacterTypeDist class, defines a character type
    def DP_ctype (self, perturbed, *args):
        
        self.DP_sample_type(perturbed, *args)
        print(self.R, self.P)
        CT = ConceptType(self.n_strokes, self.P, self.R)

        return CharacterType(self.n_strokes, CT.part_types, CT.relation_types)
    
    
    #altered function from CharacterModel class, sampling the character type of the image (model) that will be generated
    def DP_stype(self, perturbed, *args):
         return self.DP_ctype(perturbed, *args)
     
        
 ############################# GENERATE ALPHABET FUNCTION ###################################################

     ##COMEÇAR A MUDAR ESTA FUNÇÃO!!!!!  
    def optimize_generate_alphabet(self, perturbed, *args, nb_iter):
        lr=1e-3 #default learning rate 
        eps=1e-4 #default tolerance for constrained optimization
        c_type = self.DP_stype(perturbed, *args)
        self.optimize(c_type,lr, nb_iter, eps)
        return c_type
    
    
    def generate_alphabet(self, perturbed, *args, n_characters, n_tokens, nb_iter):
        
        self.p_mem(perturbed, *args)
        path = '/Users/carolinacaramelo/Desktop/alphabet'
        os.makedirs(path)
        file = open("/Users/carolinacaramelo/Desktop/alphabet/info", 'w') 
        
        
        
        columns = 6
        rows = n_characters // columns + (n_characters % columns > 0)
        #fig ,ax= plt.subplots(nrows=rows,ncols=columns,figsize=(105,105))
        fig = plt.figure(figsize=(105,105))

        count_images=0
        
        for i in range(n_characters):
            
            path_character = '/Users/carolinacaramelo/Desktop/alphabet/character%i'%i
            os.makedirs(path_character)
            c_type = self.DP_stype(perturbed, *args)
            #c_type = self.optimize_generate_alphabet(perturbed, *args, nb_iter=nb_iter)
            
            self.display_type(c_type)
            file.write("character" + str(i) + "\n")
            file.write("num strokes:" + str(c_type.k)+ "\n")
            for l in range(c_type.k):
                file.write("Stroke #:" + str(l)+ "\n")
                file.write("\tsub-stroke ids:" + str(list(c_type.part_types[l].ids.numpy()))+ "\n")
                file.write("\\trelation category:" + str(c_type.relation_types[l].category)+ "\n")
            file.write("----END CHARACTER TYPE INFO----" + "\n")
        
            
            #saves different tokens of the same character
            for j in range(n_tokens):
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                
                plt.rcParams["figure.figsize"] = [105, 105]
                plt.imsave('/Users/carolinacaramelo/Desktop/alphabet/character%i' %i + '/char%s.' %j+".png", c_image, cmap='Greys')
              
        
           
            try:
                count_images+=1
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                fig.add_subplot(rows,columns,count_images)
                plt.imshow(c_image,cmap='Greys')
                
            except:
                pass           
            
        fig.tight_layout()
        fig.savefig('/Users/carolinacaramelo/Desktop/alphabet/alphabet.png', cmap='Greys') 
        file.close()
            
            
     
        
   
    #generate alphabet will generate a set of character images related to each other - one alphabet 
    #the function will generate an alphabet with the unperturbed model and one alphabet for the perturbed model
    #the generation of the alphabets for both cases will use the same number of strokes, same number of sub-strokes and the the same primitives 
    #each alphabet will have 30 characters 
    #the sub-strokes/primitives used will be transformed into images and stores in order to have a better comprehension of what these are 
    #the alphabets will be stored in a folder 
    #20 tokens of each one of the character types 
    #1 alphabet - 30 character types - 20 tokens for each character type
    def generate_alphabet_test(self,perturbed, n_strokes, nsub, n_characters, n_tokens):
       
        
        #set the parameters to sample the alphabet
        #instead of giving the n_strokes and nsub we could also randomly sample these numbers and generate more than one alphabet all at once (with different n_strokes and nsub)
        self.n_strokes = n_strokes #know we are using these number of strokes and substrokes
        self.nsub_list = nsub #we are going to use the same number of strokes and substrokes for both alphabets
        self.subparts = [10,50,200,234,245,670,896,345,800,967,1200]
       
        self.n_characters = n_characters #for the new logT function
        
        
        #generate non perturbed alphabet 
        path = '/Users/carolinacaramelo/Desktop/alphabets/nonperturbed'
        os.makedirs(path)
        for i in range(n_characters):
            
            path_character = '/Users/carolinacaramelo/Desktop/alphabets/nonperturbed/character%i'%i
            os.makedirs(path_character)
           
            c_type = self.known_stype(perturbed) #list of character types 
            self.display_type(c_type)
            #saves different tokens of the same character
            for j in range(n_tokens):
                c_token = self.CM.sample_token(c_type)
                c_image = self.CM.sample_image(c_token)
                plt.rcParams["figure.figsize"] = (105,105)
                plt.imshow(c_image, cmap="Greys")
                plt.savefig('char%s.' %j +'.png')
                char = img.open (r"/Users/carolinacaramelo/Desktop/TESE/Code/MasterThesis/pyBPL/pybpl/model/char%s." %j+".png") 
                char.save('/Users/carolinacaramelo/Desktop/alphabets/nonperturbed/character%i' %i + '/char%s.' %j, 'PNG')
        

     

     
        
     
        
     