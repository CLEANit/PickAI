import numpy as np
import torch
from torch import randn_like as torch_randn_like
from functools import partial
from itertools import cycle

from smi_func import smi_func
from MomentsComparer import MomentsComparer
from generation_network_topology import CNA



def Init_OptFunc(opt_flagf="max"):
    '''Select which optimization extrema function.'''
    if opt_flagf == "max":

        def maximize(gen_scoref, N_best_candidates):
            best_indsf = np.argsort(np.fromiter(gen_scoref, dtype=float))
            best_indsf = best_indsf[-N_best_candidates:]
            #best_indsf = gen_scoref.argsort()[-N_best_candidates:]
            return best_indsf 

        return maximize

    elif opt_flagf == "min":

        def minimize(gen_scoref, N_best_candidates):
            best_indsf = np.argsort(np.fromiter(gen_scoref, dtype=float))
            best_indsf = best_indsf[:N_best_candidates]
            #best_indsf = gen_scoref.argsort()[:N_best_candidates]
            return best_indsf 

        return minimize

    else:
        print("opt_flag is neither \"max\" or \"min\"")
        raise TypeError

## DEV: score_metric_func should /probably/ be turned into a class with
#       a "Moments Comparer" as an attribute.. This would be a more 
#       careful accounting of the moving target.
#
#from Checkpointing import Determine_fn
#fn = Determine_fn()
fn = 6
comparer = MomentsComparer(fn)
def score_metric_func(outs):
    '''Scores each individual candidate.'''
    smi = smi_func(outs)


    #smi = outs
    #smi = np.log(smi)
    #sl, nl = np.unique(smi, return_counts=True)
    #moments_loss = np.log(nl)


    #nl, sl = nlsl(smi)
    #factors = itt.starmap(???) #Starmap works works the opposite way..
    #factors = (factor_func_i(nl, sl, x) for factor_func_i in factor_funcs)
    

##  Factors

    comparer.UpdateRange(smi)
    moments_loss = comparer.ScoreCandidates(smi)

#   Geometric averages make sense for absolute optimizations.
#   Arithemetic averages make sense for loss-minimization.
#       > goal for loss-mini is 0. Can't just filter those out..
#       > perphaps arithemetic average of /relative/ loss?

    # BUG: I THINK THIS ONLY WORKS IF EVALUATING ON SINGLES.. NOT ON ITERATORS.
    #factors = filter(lambda x: x != np.nan, moments_loss)

    # BUG: what about factors of 0?
    #       > This is why standardized moments are used.
    #       >  add a little noise?
#   Guarentee no zero factors
    #factors = filter(lambda x: x != 0, factors)
    #score_metric = np.prod(np.fromiter(factors, dtype=float))
    #score_metric = score_metric**(1/N_factors)

    # BUG: what about factors of 0?
    #       > This is why standardized moments are used.
    #       >  add a little noise?
    score_metric = sum(moments_loss)
    return score_metric



create_mutation = partial(torch_randn_like, requires_grad=False)


@torch.no_grad()
def procreate(mutation_rate, N_descendants, ancestor_state_dict):
#, gen, first_empty_index):
    '''Within an generation-array, reproduce a successful `ancestor' design \
       (contained at ancestor_ind w/in 'gen') N_descentants times.'''

    # Loop over descendants_range, overwritting failed candidate w/descendants.
    #for descendant_index in range(*descendants_range):
    for descendant_index in range(N_descendants):

        # Load network to be overwritten
        ##  CHECK :: deep-copy here? :: LOOKS OKAY.
        descendant_i = CNA()
        #gen[descendant_index]
        descendant_i.load_state_dict(ancestor_state_dict)

        mutations = map(create_mutation, descendant_i.parameters())

        # Mutate each parameter
        #mutations = (mut_i*mutation_rate for mut_i in mutations)
        #mutations_param_pairs = zip(mutations, descendant_i.parameters())
        #for mutation_i, param in mutations_param_pairs:
        for param in descendant_i.parameters():
            mutation_i = next(mutations)
            mutation_i *= mutation_rate

            # Add mutation to parameter    
            param.add_(mutation_i)


        yield descendant_i
        

def ReDistribute_N_Offspring(gen_size, N_best_candidates, N_runners_up):
    N_honourable_mentions = N_best_candidates - N_runners_up

    reward_1 = (gen_size-N_best_candidates)//3#//1 frac of next gen decended from best
    reward_2 = reward_1//N_runners_up
    reward_3 = reward_1//N_honourable_mentions #=(10-3-1)

    # Hackey fix? sometimes r3 was < 1?
    reward_3 = max(1, reward_3)
    reward_1 = gen_size \
               -N_best_candidates \
               -N_runners_up*reward_2 \
               -N_honourable_mentions*reward_3

    return reward_1, reward_2, reward_3


def UpdateBatchSize(epoch, epoch_mult, max_batch_size, min_batch_size):
    #batch_size = int(2**round(np.log2(11+epoch))) #increases as gen shrinks
    batch_size = min_batch_size + ((max_batch_size - min_batch_size)/epoch_mult)*epoch
    
    # Calc nearest power of 2.
    batch_size = int(2**round(np.log2(batch_size)))

    # Enforce batch_size range
    #batch_size = max(batch_size, min_batch_size)
    #batch_size = min(batch_size, max_batch_size)
    return batch_size


def UpdateLearningRate(epoch, max_mutation_rate, diff_mutation_rate):
        mutation_rate = max_mutation_rate
        mutation_rate -= (epoch+1)*diff_mutation_rate
        return mutation_rate


def UpdateGenSize(epoch, epoch_mult, max_gen_size=1000, min_gen_size=25):
    diff_gen_size = (epoch * (max_gen_size - min_gen_size))//(epoch_mult+1)
    gen_size = max_gen_size - diff_gen_size
    gen_size = int(gen_size)
    return gen_size


def EvaluateGeneration_Base(candidate, batch_all):
    candidate_score = map(candidate, batch_all)
    candidate_score = map(score_metric_func, candidate_score)

    # Summing across all batches
    candidate_score = sum(candidate_score)
    #candidate_score /= len(batch_all)
    return candidate_score



#class ReGenerator():
#    def __init__(self, 

def ReGenerate_Gen(mutation_rate, N_best_candidates, N_runners_up, gen_size, rewards, best_candidates):
                   #gen, best_inds):

    reward_1, reward_2, reward_3 = rewards

    # Can't be lazy eval'd here. ??? FIXED?
   
    # NEEDS TO BE IN THE MAIN SCRIPT TO THAT BEST OF GEN CAN BE CHECKPOINTED`
    #best_candidates = (gen[i] for i in best_inds) 
    #best_candidates = cycle(best_candidates)

    #for reg_index, fit_candidate in enumerate(best_candidates):
    #    try:
    #        gen[reg_index].load_state_dict(fit_candidate.state_dict())

    #    except RuntimeError as e:
    #        print(e)
    #        gen[reg_index] = fit_candidate
    #
    ##

    #  Populate
    first_empty_index = N_best_candidates
    for best_candidate_ind in range(N_best_candidates):
        best_candidate_i = next(best_candidates)
        yield best_candidate_i


    best_candidates_sd = (candi.state_dict() for candi in best_candidates)

    # Best candidate get r1 decentants
    ancestor_sd = next(best_candidates_sd)
    yield from procreate(mutation_rate,
                        reward_1,
                        ancestor_sd)
    first_empty_index += reward_1
    
    # Next 3 best get r2 decendants
    for ancestor_index in range(1, N_runners_up):
        ancestor = next(best_candidates_sd)
        yield from procreate(mutation_rate,
                             reward_2,
                             ancestor_sd)
        first_empty_index += reward_2
    
    # Rest of the results get divied up among rest.
    #   > Looping back over entire set of best candidates to deal w/ residuals
    while first_empty_index < gen_size:
        # Don't exceed amount of space left in generation
        #   > DEV: COULD AVOID NEEDING TO LOOP IF REWARD_1 WAS LEFTOVERS FROM
        #   @ DEV2: REWARD 1 IS DEFINED THIS WAY NOW....?? 
        #          ``reward_1 = gen_size - 3*reward_2 - 6*reward_3''
        new_reward = min(reward_3, gen_size - first_empty_index)

        # Get ancestor_i for reward (loops)
        ancestor_sd = next(best_candidates_sd)

        # product iterator from ancestor_i
        yield from procreate(mutation_rate,
                             new_reward,
                             ancestor_sd)
    
        first_empty_index += new_reward

        # descendants is an array of iterators each coming from a
        # previous sucessful candidate
        #yield from gen_new
    

