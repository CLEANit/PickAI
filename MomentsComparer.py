from smi_func import smi_func
from itertools import starmap
import torch

# DEV ADD CHECK-POINT UPDATER OF MAX/MIN
class MomentsComparer():
    def __init__(self, fn):
        self.fn = fn
        self.InitRandomProperties()


    def GetRandomSample(self, sample_size=1000):
        from numpy.random import randint
        sample = randint(0,2,(sample_size, 1, self.fn, self.fn))
        return sample


    def GetRandomDist(self):
        sample = self.GetRandomSample()
        property_sample = smi_func(sample)
        return property_sample


    def InitRandomProperties(self):
        '''Without exploration, the limits of the property-space \
           are unknown. Begin with a random sample. \
           Has the risk of trucating / minimizing the actual domain'''

        random_sample = self.GetRandomDist()
        
        sample_max = random_sample.max()
        sample_min = random_sample.min()

        self.max = sample_max
        self.min = sample_min
        
        # Once initial range is estimated. Estimate ideal moments.
        self.CalculateIdealMoments()


    @classmethod 
    def CalculateMoments(self, sample_max, sample_min):
        '''With a known distribution, [here default is uniform] \
           the "correct" values for central moments can be \
           calculated from max and min.'''

        # First Central Moment (mean)
        mean = sample_max + sample_min
        mean /= 2

        # Second Central Moment (var)
        #var = (1/12) * (sample_max - sample_min)**2
        n = (sample_max - sample_min + 1)
        var = (n**2 - 1)/12

        # Third central moment
        skew = 0

        # Fourth central moment
        #n = sample_max - sample_min + 1
        ex_kurtosis = (6 * (n**2 + 1)) / (5 * (n**2 - 1))

        #wiki: /excess kurtosis/ is defined as kurtosis minus 3

        return mean, var, skew, ex_kurtosis

    def CalculateIdealMoments(self):
        
        self.ideal_moments = self.CalculateMoments(self.max, self.min)
        #self.ideal_moments = torch.FloatTensor(self.ideal_moments)

        #self.ideal_mean = self.ideal_moments[0]
        #self.ideal_var =  self.ideal_moments[1]
        #self.ideal_skew = self.ideal_moments[2]
        #self.ideal_kurt = self.ideal_moments[3]

    @classmethod 
    def EstimateMoments(self, candidate_sample):
        ## Scipy.stats has a moment-calculating function.
        #       > But is it an unbiased estimator?

        len_sample = candidate_sample.__len__()
        sample_mean = candidate_sample.mean()

        # Value reused several times.
        candidate_difference = candidate_sample - sample_mean

        sample_var = candidate_difference**2
        sample_var = sample_var.sum()
        #try:
        #    assert sample_var != 0

        #except AssertionError:
        #    #yield sample_mean, 0, 0, 0
        #    return sample_mean, 0, 0, 0


        bias_var = (1/len_sample) * sample_var
        sample_var *= (1/(len_sample - 1)) 
        
        # population skew numerator
        sample_skew = candidate_difference**3
        sample_skew = sample_skew.sum()
        sample_skew /= len_sample
        # population skew denominator
        sample_skew /= sample_var**(3/2)

        # sample estimator adjustment for skew
        sample_skew *= len_sample**2 / ((len_sample - 1)*(len_sample - 2))


        # KURT.
        # kurt_numerator
        sample_kurt =  candidate_sample**4
        sample_kurt = sample_kurt.sum()
        sample_kurt *= 1/len_sample

        #kurt denomenator
        sample_kurt /= bias_var**2
        sample_kurt -= 3

        # sample estimator adjustment for kurt
        sample_kurt *= len_sample + 1
        sample_kurt += 6
        sample_kurt *= (len_sample - 1) / ((len_sample - 2)*(len_sample - 3))

        return sample_mean, sample_var, sample_skew, sample_kurt
        #yield sample_mean, sample_var, sample_skew, sample_kurt


    def ScoreCandidates(self, candidate_smi):

        candidate_moments = self.EstimateMoments(candidate_smi)

        # recall that internal max and min should be updated after each 
        # production... if not.. it'll have to be done here.
        #self.UpdateRange(candidate_smi)

        expected_vs_predicted = zip(candidate_moments, self.ideal_moments)
        moments_loss = starmap(self.LossFunc, expected_vs_predicted)

        #moments_loss = (self.LossFunc(
        #                 candidate_moments[moment_ind],self.ideal_moments[moment_ind]
        #               )
        #                         for moment_ind in range(len(self.ideal_moments)))

        #moments_loss = np.prod(np.fromiter(moments_loss))
        return moments_loss

    
    def LossFunc(self, x_estimate, x_true):
        loss = abs(x_true - x_estimate)
#        if x_true != 0:
#            #loss = ((x_true - x_estimate)/x_true)**2
#            #loss = (1 - x_estimate/x_true)**2
#            loss = abs((1 - x_estimate/x_true))
#
#        else:
#            #x_true == 0
#            loss = abs(x_estimate)
#
#            #loss = (x_true - x_estimate)**2
#
        return loss


    def UpdateRange(self, candidate_sample):
        try:
            candidate_max = candidate_sample.max()
            candidate_min = candidate_sample.min()

        except AttributeError:
            candidate_max = max(candidate_sample)
            candidate_min = max(candidate_sample)

       
        #init flag.
        update_flag = False
        if candidate_max > self.max:
            self.max = candidate_max
            update_flag = True

        if candidate_min < self.min:
            self.min = candidate_min
            update_flag = True

        if update_flag == True:
            self.CalculateIdealMoments()



# Old experimental factors
factor_funcs = (lambda nl, sl, x: (len(sl))**3,
                #drives up min(nl), drives down max(nl)
                lambda nl, sl, x: abs(min(nl)/max(nl))**2,
                #e.g. 72 -(-72) -> makes large & symm.
                lambda nl, sl, x: abs(max(sl) - min(sl)),
                #lambda nl, sl, x: abs(max(sl)*min(sl)),
                # Minimizes difference between agacents
                #lambda nl, sl, x: 2**(-int(max(np.diff(sl)))),
                # Maximizes spread in positive dim.
                #lambda nl, sl, x: np.dot((np.abs(sl)+sl),nl),
                # Maximizes spread in negative dim.
                #lambda nl, sl, x: np.dot((np.abs(sl)-sl),nl),
                # Maximizes diversity of inputs
                lambda nl, sl, x: str_split(x),
                #
                # DEV: depends on proof that product of a finite splits
                #      is maximized when segments are equal
                lambda nl: np.sum(np.log(nl))
)
N_factors = len(factor_funcs)

