import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, PandasTools, Descriptors
from functools import partial
from dgllife.utils import smiles_to_bigraph
from dgllife.utils.featurizers import BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer, atom_explicit_valence_one_hot
from dgllife.utils.featurizers import atom_total_degree_one_hot, atom_formal_charge_one_hot, atom_is_aromatic, atom_type_one_hot
from dgllife.utils.featurizers import atom_implicit_valence_one_hot, bond_type_one_hot, bond_is_in_ring
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score
import dgl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import R2Score
from copy import deepcopy
from dgllife.model.model_zoo import AttentiveFPPredictor


"""
Functions for molecule graph creation and featurization.
"""
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def chirality(atom):
    """
    Function for adding chirality information to node featurizer.
    """
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

    
def construct_graph_and_featurize(smiles_string, add_self_loop=False, canonical_atom_order=False, 
                                  explicit_hydrogens=False):
    """
    Function for converting a SMILES string into a DGL bigraph and featurizing its node and edges.
    """
    atoms = ['H','N','O','C','P','S','F','Br','Cl','I','Si']

    atom_total_degrees = list(range(6))
    atom_formal_charges = [-1, 0, 1]
    atom_implicit_valence = list(range(4))
    atom_explicit_valence = list(range(8))

    atom_concat_featurizer = ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=atoms), 
                                                partial(atom_total_degree_one_hot, 
                                                        allowable_set=atom_total_degrees),
                                                partial(atom_formal_charge_one_hot, 
                                                        allowable_set=atom_formal_charges),
                                                atom_is_aromatic,
                                                partial(atom_implicit_valence_one_hot, 
                                                        allowable_set=atom_implicit_valence),
                                                partial(atom_explicit_valence_one_hot, 
                                                        allowable_set=atom_explicit_valence),
                                                chirality])

    atom_featurizer = BaseAtomFeaturizer({'atom': atom_concat_featurizer})

    # Bond featurizer for stage 1
    bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
    bond_featurizer = BaseBondFeaturizer({'bond': bond_concat_featurizer})
    
    graph = smiles_to_bigraph(
        smiles_string, add_self_loop=add_self_loop, node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer, canonical_atom_order=canonical_atom_order, 
        explicit_hydrogens=explicit_hydrogens)

    return graph


def dataset_from_smiles_label_tups(list_o_tups):
    
    dataset = list()
    for tup in list_o_tups:
        graph = construct_graph_and_featurize(tup[0])
        label = tup[1]
        dataset.append((graph,label))
    
    return dataset


# make sure we choose the same test sets every time
def create_task_datasets(kinase_df, target_kinase, exclude_target_smiles=True,
                         max_aux_task_size=4000, random_seed=None):
    """
    Function for creating the individual task datasets for meta-learning
    """
    if random_seed:
        np.random.seed(random_seed)
    else:
        np.random.seed(43) # seed I've been using

    # exclude any smiles from the pKi sample of the target kinase from pre-training data
    if exclude_target_smiles:
        target_df = kinase_df[(kinase_df['Kinase_name']==target_kinase) &\
                                   (kinase_df['measurement_type']=='pKi')]
        exclusion_list = target_df.SMILES.values.tolist()
        exclusion_dict = {smiles:0 for smiles in exclusion_list}

    kinases = kinase_df.Kinase_name.unique().tolist()
    measures = kinase_df.measurement_type.unique().tolist()

    datasets_dict = dict()

    for kinase in kinases:
        for measure in measures:
            subset_df = kinase_df[(kinase_df['Kinase_name']==kinase) &\
                                  (kinase_df['measurement_type']==measure)]
            smiles_labels = subset_df[['SMILES','measurement_value']].values.tolist()

            if exclude_target_smiles and (not kinase == target_kinase):
                dataset = list()
                for smiles, label in smiles_labels:
                    # don't include any molecules from our target in pre-training
                    if smiles in exclusion_dict:
                        continue
                    graph = construct_graph_and_featurize(smiles)
                    label = torch.Tensor([label]) 
                    dataset.append((graph, label))

                np.random.shuffle(dataset)
                dataset = dataset[:max_aux_task_size]

            else:
                dataset = list()
                for smiles, label in smiles_labels:
                    graph = construct_graph_and_featurize(smiles)
                    label = torch.Tensor([label]) 
                    dataset.append((graph, label))

                np.random.shuffle(dataset)

            # get some test set indices
            if (measure == 'pKi') and (kinase == target_kinase):
                dt_len = len(dataset)
                test_indices = np.random.choice([i for i in range(dt_len)], 
                                            int(np.ceil(dt_len*0.2)), replace=False)

                train_data = list()
                test_data = list()
                for idx, tup in enumerate(dataset):
                    if idx in test_indices:
                        test_data.append(tup)
                    else:
                        train_data.append(tup)

                datasets_dict[kinase+'-'+measure] = dict()
                datasets_dict[kinase+'-'+measure]['train'] = train_data
                datasets_dict[kinase+'-'+measure]['test'] = test_data
            else:
                datasets_dict[kinase+'-'+measure] = dict()
                datasets_dict[kinase+'-'+measure]['train'] = dataset
            
    return datasets_dict
        
    
class MetaDataClass:
    """
    Class for constructing the datasets needed for meta-learning training
    and hyper-parameter tuning. In this context we are using the pKi training
    data of a task of interest as a test/validation set for tuning. The holdout
    will not be used until final evaluation.
    """
    def __init__(self, datasets_dict, target_kinase=None, meta_task_bs=100,  
                    train_bs=10, target_task_subset=200, random_seed=43):
        
        self.current_task = 0 
        self.meta_task_batch_size = meta_task_bs # how many train samples per epoch
        self.train_batch_size = train_bs # how many train samples per train batch
        self.target_task = target_kinase+'-pKi'
        self.aux_task_list = list()
        self.target_task_subset = target_task_subset # limit number of samples for validation
        self.task_dict = datasets_dict
        self.aux_task_idx_dict = dict()
        self._init_aux_tasks_idx_dict()
        

    def _init_aux_tasks_idx_dict(self):
        
        # add aux tasks to idx dict
        for task in self.task_dict:
            if task == self.target_task:
                continue
            self.aux_task_list.append(task)
            self.aux_task_idx_dict[task] = dict()
            self.aux_task_idx_dict[task]['current_idx'] = 0
            self.aux_task_idx_dict[task]['total_len'] = len(self.task_dict[task]['train'])
        
        
    def get_aux_task_train_dataloader(self):
        # function for getting a train batch from the current aux tasks
        current_aux_task = self.aux_task_list[self.current_task]
        current_task_idx = self.aux_task_idx_dict[current_aux_task]['current_idx']
        current_batch = self.task_dict[current_aux_task]['train']\
                     [current_task_idx:current_task_idx+self.meta_task_batch_size]
        
        batch_len = len(current_batch)
        if not batch_len == self.meta_task_batch_size:
            # update current task idx
            num_needed = self.meta_task_batch_size-batch_len
            current_batch += self.task_dict[current_aux_task]['train'][:num_needed]
            self.aux_task_idx_dict[current_aux_task]['current_idx'] = num_needed
        else:
            # update current task idx
            self.aux_task_idx_dict[current_aux_task]['current_idx'] = current_task_idx+\
                                                               self.meta_task_batch_size
            
        # update meta task idx
        next_task = (self.current_task+1)%len(self.aux_task_list)
        self.current_task = next_task
       
        dataloader = dgl.dataloading.GraphDataLoader(current_batch, batch_size=10, 
                                    shuffle=True, drop_last=False)
        
        return dataloader
    
    
    def get_target_task_dataloader(self):
        # function for getting a test batch
        target_batch = self.task_dict[self.target_task]['train'][:self.target_task_subset]
        target_dataloader = dgl.dataloading.GraphDataLoader(target_batch, batch_size=10, 
                                    shuffle=True, drop_last=False)
        
        return target_dataloader

    
def meta_pretrain_model(model, mt, innerstepsize = 0.00001,
                       innerepochs = 1, outerstepsize0 = 0.0001, 
                        niterations = 10000, seed=0):
    """ 
    Function for model pre-training using a meta-learning approach 
    """
    #mt - meta-task training object
    #innerstepsize - stepsize in inner SGD
    #innerepochs - number of epochs of each inner SGD
    #outerstepsize0 - stepsize of outer optimization, i.e., meta-optimization
    #niterations - number of outer updates; each iteration we sample one task and update on it

    torch.manual_seed(seed)
    r2score = R2Score()

    #loss_criterion = nn.L1Loss() # good results with this
    loss_criterion = nn.MSELoss(reduction='none')

    # Reptile training loop
    for iteration in range(niterations):
        weights_before = deepcopy(model.state_dict())
        # Generate task
        task_dataloader = mt.get_aux_task_train_dataloader()
        # Do SGD on this task

        train_losses = list()

        for _ in range(innerepochs):
            for batch_id, batch_data in enumerate(task_dataloader):
                bg, labels = batch_data
                labels = labels.reshape(-1,1)
                prediction = model(bg, bg.ndata['atom'], bg.edata['bond'])
                #loss = (loss_criterion(prediction, labels)*(masks != 0).float()).mean()
                loss = loss_criterion(prediction, labels).mean()
                loss.backward()
                for param in model.parameters():
                    param.data -= innerstepsize * param.grad.data
                train_losses.append(loss.data.item()) 

        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights_after = model.state_dict()
        outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
        model.load_state_dict({name : 
            weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
            for name in weights_before})

        # Periodically test model
        if (iteration+1) % 100 == 0:
            print('Traning on validation set:')
            weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
            target_dataloader = mt.get_target_task_dataloader()

            for inneriter in range(100):
                test_losses = list()
                r2_scores = list()
                for batch_id, batch_data in enumerate(target_dataloader):
                    bg, labels = batch_data
                    labels = labels.reshape(-1,1)
                    prediction = model(bg, bg.ndata['atom'], bg.edata['bond'])
                    #loss = (loss_criterion(prediction, labels)*(masks != 0).float()).mean()
                    loss = loss_criterion(prediction, labels).mean()
                    loss.backward()
                    for param in model.parameters():
                        param.data -= innerstepsize * param.grad.data
                    test_losses.append(loss.data.item()) 
                    r2_scores.append(r2score(prediction.detach(), 
                                                  labels.detach()))

                if (inneriter+1) % 8 == 0:
                    total_score = np.mean(test_losses)
                    mean_r2 = np.mean(r2_scores)
                    print('Test epoch: {:d}/{:d}'.format(inneriter + 1, 32))
                    print('Avg loss: {:.4f}'.format(total_score))
                    print('Avg R2: {:.4f}'.format(mean_r2))
                    print('')

            model.load_state_dict(weights_before) # restore from snapshot

        if (iteration+1)%100 == 0:
            print(iteration+1,'iteration complete...')

        #print(np.mean(train_losses))
    final_weights = deepcopy(model.state_dict())
    return final_weights
