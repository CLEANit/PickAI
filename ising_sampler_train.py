import torch
import numpy as np
import functools as ft
from itertools import cycle
import time
import os
import sys

sys.path.append(os.getcwd())
from generation_network_topology import CNA
from input_func import *
from smi_func import smi_func #smi_func is the Hamiltonian
from datafuncs import *

import GA_Tools as GA
from MomentsComparer import MomentsComparer


try:
    fn = int(sys.argv[1])
except:
    fn = 6

import Train_Checkpointing as checkpoint
input_width = checkpoint.DetermineInputSize(fn)    


with torch.no_grad():    
    
    # Optimization parameters
    opt_flag = "max"
    epoch_mult = 1024 
    
    # Evolution parameters
    N_runners_up = 4
    N_best_candidates = 10

    max_mutation_rate = 1.0
    min_mutation_rate = 0.0001

    max_gen_size = 250
    min_gen_size = 25

    diff_mut_rate = (max_mutation_rate-min_mutation_rate)
    diff_mut_rate /= (epoch_mult+1)
    
    # Efficiency parameters
    batch_mult = 16
    max_batch_size = 512
    min_batch_size = 32


    # Below: round over corrects...
    #diff_gen_size = int(diff_gen_size)


##  Generation Initialization
    epoch, gen_size, \
    batch_size, cp_dict, \
    gen = checkpoint.ReturnToCheckpoint(epoch_mult,
                                max_gen_size,
                                min_gen_size
    )
    gen = list(gen)

    gen_size = GA.UpdateGenSize(epoch,
                                epoch_mult,
                                max_gen_size,
                                min_gen_size)

    opt_func = GA.Init_OptFunc(opt_flag)

##  Train
    while epoch < epoch_mult:
        t_epoch = time.time()

        # contines for N mins. Then saves.
        while time.time() - t_epoch < 5*60:   
            epoch += 1

            # Adjust generation size
            gen_size = GA.UpdateGenSize(epoch,
                                        epoch_mult,
                                        max_gen_size,
                                        min_gen_size)

            # Different reward levels (depending on gen_size)
            rewards = GA.ReDistribute_N_Offspring(
                                          gen_size,
                                          N_best_candidates,
                                          N_runners_up)

            # Learning rate moves from maximum to minimum 
            # allowed over epoch_mult
            mutation_rate = GA.UpdateLearningRate(
                                        epoch,
                                        max_mutation_rate,
                                        diff_mut_rate)
            
            # Batch size increases as generation size decreases 
            #   (more testing per candidate near end)
            batch_size = GA.UpdateBatchSize(epoch,
                                            epoch_mult, 
                                            max_batch_size,
                                            min_batch_size
            )

            # Pre-generate all batches. 
            batch_shape = (batch_mult,
                           batch_size,
                           1,
                           input_width,
                           input_width)

            batch_all = input_func_all(batch_shape)

            # Load all-batches into partial function for lazy 
            # evaluation.
            EvaluateGeneration=ft.partial(
                                  GA.EvaluateGeneration_Base,
                                  batch_all=batch_all
            )

            # Lazy evaluation of candidate score
            # (does all candidates in gen)
            gen_score = map(EvaluateGeneration, gen)

            ##  Score
            best_inds = opt_func(gen_score,
                                 N_best_candidates)

            ##  Sort
            best_candidates = (gen[i] for i in best_inds)
            best_candidates = cycle(best_candidates)

            new_gen = GA.ReGenerate_Gen(mutation_rate,
                                        N_best_candidates,
                                        N_runners_up, 
                                        gen_size,
                                        rewards,
                                        best_candidates)
            # Regenerate.
            gen = list(new_gen)
            median_ind = N_best_candidates//2
            median_best_score = EvaluateGeneration(
                                    gen[median_ind]
            )

            print(median_best_score,
                  f"{gen_size}/{len(gen)}",
                  batch_size)


        # Report progress
        candidate_sample = map(gen[N_best_candidates//2],
                               batch_all)

        candidate_sample = map(
                            MomentsComparer.EstimateMoments,
                            candidate_sample
        )
        
        moments = next(candidate_sample)
        moments = [round(mmnt_i.item(), 3) \
                     for mmnt_i in moments]
        print(f"median best moments: {moments}")
        
        # Compare against known range.
        ideal_moments = MomentsComparer.CalculateMoments(72,
                                                         -72)
        print(f"ideal: {ideal_moments}")
        
        # Save optimization
        checkpoint.MakeCheckpoint(epoch, epoch_mult, input_width,
                          best_candidates, N_best_candidates,
                          max_mutation_rate, diff_mut_rate, 
                          gen_size, fn,
                          batch_mult, batch_size)

                
# Final optimization. Shrinks gen to only best
for i, gen_i in enumerate(gen[:N_best_candidates]):
    print(f"\ngen[{i}]")
    input_batch = input_func_all((50000,
                                  1,
                                  input_width,
                                  input_width))
    outs = (gen_i(input_batch)).detach()
    smi = smi_func(outs)

    sl, nl = np.unique(smi, return_counts=True)
    print("sl: ", sl)
    print("nl: ", nl)
    print("score: ", GA.score_metric_func(outs))
    print("test: ", smi_func(outs)[0])
    
for i in range(N_best_candidates):
    torch.save(gen_i.state_dict(),
               f"candidate_networks/inet{fn}_{i}.pt")
print("FIN")

