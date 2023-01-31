import numpy as np
class lp():
    def __init__(self, nval=10):
        # initialize array
        self.best = [9]*nval

    def LazyPartition(self, itr):
        for val in itr:
            # does belong on list?
            #if val > self.best[0]:
                # recursive sort.
            self.RecurrsiveSort(val, 0)


            print(self.best)
    
        return self.best

    def RecurrsiveSort(self, val, i):
        print(val, i, self.best)
        if val >= self.best[i]:
            try:
                self.RecurrsiveSort(val, i+1)
    
            except IndexError:
                print(f"overshot: {val} :: {self.best}")
                self.best = self.best[1:i+1]+[val] + self.best[i+1:]
                #self.best = self.best[1:]+[val]
                #self.best = self.best[1:i+1]+[val] + self.best[i+1:]
        else:
            try:
                assert i > 0
                print(f"found: {val} :: {self.best}")
                raise IndexError

    
            except AssertionError:
                pass
    
    

def ExpensiveFunc(x):
#    time.sleep(1)
    return x


import random
itr = range(20)
itr = list(itr)
random.shuffle(itr)
ten_ind = itr.index(10)
itr.pop(ten_ind)
itr.append(10)

lp = lp()
best = lp.LazyPartition(itr)
print(best)

task = map(ExpensiveFunc, range(20))

from GA_Tools import opt_func
g = opt_func(task, 10)


