import torch
import json
import os
import sys

sys.path.append(os.getcwd())
from smi_func import smi_func #smi_func is the Hamiltonian
import datafuncs
import GA_Tools as GA
from input_func import input_func_all


def DetermineInputSize(fn, max_input_width=1000):

    from generation_network_topology import CNA
    mod = CNA()
    
    #while searching_flag == True:
    input_width = 1
    while input_width < max_input_width:
        input_width += 1

        dummy_input = input_func_all((10, 1, input_width, input_width), grad_req=True)
        try:
            out_test = mod.MakeConfig(dummy_input)
        except RuntimeError:
            continue
        except IndexError:
            continue

        out_width = out_test.shape[-1]
        if out_width == fn:
            return input_width 
            #break
    
        else:
            continue

    # Error out if this point is reached.
    raise ValueError




##  Checkpointing

def ReturnToCheckpoint(epoch_mult, max_gen_size, min_gen_size):
    from generation_network_topology import CNA
    # Attempt to return to checkpoint.
    try:
        with open("checkpoint.json", 'r') as fil:
            checkpoint_dict = json.load(fil)

        epoch = checkpoint_dict["epoch"]
        #gen_size = checkpoint_dict["gen_size"]
        gen_size = GA.UpdateGenSize(epoch, epoch_mult, max_gen_size, min_gen_size)

    except FileNotFoundError:
        print("checkpoint not found")
        epoch = 0
        gen_size = GA.UpdateGenSize(epoch, epoch_mult, max_gen_size, min_gen_size)
        checkpoint_dict=dict()


    #Check if networks exist or not. if so add them to the next generation.
    directory_contents = os.listdir("./candidate_networks/")
    checkpointed_nets = filter(lambda fil: fil[-3:] == ".pt", directory_contents)

    gen = []
    for checkpointed_net_i in checkpointed_nets: 
            # make a new net 
            net = CNA()

            # load checkpointed_net
            net.load_state_dict(torch.load(f"candidate_networks/{checkpointed_net_i}"))

            #with open(f"candidate_networks/inet{fn}_{candidate_index}.pkl", 'rb') as fil:
            #with open(f"candidate_networks/{checkpointed_net_i}", 'rb') as fil:
            #    net = pickle.load(fil)

            gen.append(net)

    print(f"{len(gen)} networks found")
    
    N_new_recruits = gen_size - len(gen)
    for i in range(N_new_recruits):
        gen.append(CNA())

    batch_size = GA.UpdateBatchSize(epoch, epoch_mult, 128, 16)

    return epoch, gen_size, batch_size, checkpoint_dict, gen


def MakeCheckpoint(epoch, epoch_mult, input_width, best_candidates, N_best_candidates, max_mutation_rate, diff_mutation_rate, gen_size, fn, batch_mult, batch_size): 

    mutation_rate = GA.UpdateLearningRate(epoch,
                                          max_mutation_rate,
                                          diff_mutation_rate)
    batch_size = GA.UpdateBatchSize(epoch, epoch_mult, 128, 16)
    batch = input_func_all((100, 1, input_width, input_width))

    candidate = next(best_candidates)
    outs = candidate(batch) #Pick a representative from gen
    print("~~~~~~~~~")
    print("epoch: ", epoch)
    print("gen_size: ", gen_size)
    print("mutation_rate: ", mutation_rate)
    print(outs[0])

    smi_prnt = smi_func(outs)
    nl, sl = datafuncs.nlsl(smi_prnt)
    print("nl: ",nl)
    print("sl: ",sl)

    score_metric = GA.score_metric_func(outs)
    print("score metric: ", int(score_metric))
#
##

##  Saving
#
    #for candidate_index, candidate in enumerate(gen[:N_best_candidates]):
    for candidate_index in range(1, N_best_candidates):
        torch.save(candidate.state_dict(), f"candidate_networks/inet{fn}_{candidate_index}.pt")
        #with open(f"candidate_networks/inet{fn}_{candidate_index}.pkl", 'wb') as fil:
        #    pickle.dump(candidate, fil)

        candidate = next(best_candidates)
#
##

##   Documenting
#
    checkpoint_dict = {}
    checkpoint_dict["epoch"] = epoch
    checkpoint_dict["nl"] = nl.tolist()
    checkpoint_dict["sl"] = sl.tolist()
    checkpoint_dict["mutation_rate"] = mutation_rate
    checkpoint_dict["gen_size"] = gen_size
    checkpoint_dict["input_eg"] = batch[0].numpy().tolist()
    checkpoint_dict["output_eg"] = outs[0].numpy().tolist()
    checkpoint_dict["fn"] = fn
    checkpoint_dict["batch_size"] = batch_size
    checkpoint_dict["batch_mult"] = batch_mult

    with open("checkpoint.json", 'w') as fil:
        json.dump(checkpoint_dict, fil)

