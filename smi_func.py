from Hamiltonian import Hamiltonian, Hamiltonian_multi
def smi_func(x):
    try:
        smi_sample = x.squeeze().numpy()

    except AttributeError:
        # Hamiltonian is coming out right now.
        smi_sample = Hamiltonian_multi(x)


    #try:
    #    #smi_sample = list(map(Hamiltonian, x))#[Hamiltonian(xi) for xi in x]
    #    #smi_sample = np.array(smi_sample)
    #    #return map(Hamiltonian, x)#[Hamiltonian(xi) for xi in x]
    #except np.AxisError:
    #    smi = Hamiltonian(x)

    return smi_sample
