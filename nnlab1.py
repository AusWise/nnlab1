'''
Created on Oct 4, 2015

@author: auswise
'''

import numpy
import matplotlib.pyplot as plt

class Neuron(object):
    def __init__(self, w, theta=0, bipolar=False):
        self.w = w
        self.theta = 0
        self.bipolar = bipolar
        
    def f(self, x):
        x = _add_bias(x)
        if len(x)!=len(self.w):
            raise Exception()
        return self._f(x)
    
    def _f(self, x):
        pos = 1
        if self.bipolar:
            neg = -1
        else:
            neg = 0
        s = numpy.dot(self.w, x)
        if s>=self.theta:
            return pos
        else:
            return neg
        
    def delta(self, x, y):
        pass
    
    def learn(self, patterns, alpha, eps):
        i = 0 
        for p in _permute(patterns):
            print str(i) + ": " + str(self.w)
            for (x, y) in p:
                d = self.delta(x, y)
                x = _add_bias(x)
                self.w = self.w + numpy.multiply(alpha*d, x)
            
            if self._stop_condition(p, eps):
                return i
            
            i += 1    
                
        return i
    
    def _stop_condition(self, patterns, eps):
        pass

class Perceptron(Neuron):
    def __init__(self, w, theta=0, bipolar=False):
        super(Perceptron, self).__init__(w, theta, bipolar)
        
    def delta(self, x, y):
        return y - self.f(x)
    
    def _stop_condition(self, patterns, eps):
        for (x, y) in patterns:
            if abs(self.delta(x, y))>eps:
                return False
        return True
                 
class Adaline(Neuron):
    def __init__(self, w, theta=0, bipolar=False):
        super(Adaline, self).__init__(w, theta, bipolar)
        
    def delta(self, x, y):
        x = _add_bias(x)
        return y - numpy.dot(self.w, x)
    
    def _stop_condition(self, patterns, eps):
        error = 0
        for (x, y) in patterns:
            error += self.delta(x, y)**2
        
        N = len(patterns)
        error = error/N
        return error<eps    
    
def _add_bias(x):
    return [1] + x

def _permute(xs, low=0):
    if low+1>=len(xs):
        yield xs
    else:
        for p in _permute(xs, low+1):
            yield p
        for i in range(low+1, len(xs)):
            xs[low], xs[i] = xs[i], xs[low]
            for p in _permute(xs, low+1):
                yield p
            xs[low], xs[i] = xs[i], xs[low]
    

def _analyze_once(alpha, a, eps, bipolar=False, adaline=False, theta=0):
    if bipolar:
        patterns = [
            ([0,0],-1), 
            ([0,1],-1), 
            ([1,0],-1), 
            ([1,1],1)
            ]
    else:
        patterns = [
            ([0,0],0), 
            ([0,1],0), 
            ([1,0],0), 
            ([1,1],1)
            ]
    w = numpy.random.rand(2+1)
    w = numpy.multiply(2*a, w)
    w[:] = [wi - a for wi in w]
    if adaline:
        neuron = Adaline(w, bipolar=bipolar)
    else:
        neuron = Perceptron(w, bipolar=bipolar)
    t = neuron.learn(patterns, alpha, eps)
    w = neuron.w
    print w 
#     print t
#     for (x,y) in patterns:
#         print str(neuron.f(x)) + " " + str(neuron.delta(x, y))
#     w0 = w[0]
#     w1 = w[1]
#     w2 = w[2]
#     x2 = lambda x1: -w1/w2*x1 - w0/w2
#     plt.plot([-0.1,1.1],[x2(-0.1), x2(1.1)])
#     plt.plot([0,1], [[0,1],[0,1]], "ro")
#     plt.plot(1,1, "go")
#     plt.axis([-0.1,1.1,-0.1,1.1])
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.suptitle("")
#     title = "from=" + str(-a) + " to=" + str(a) + " alpha=" + str(alpha)
#     filename = "fig_from=" + str(-a) + "_to=" + str(a) + "_alpha=" + str(alpha)
#     if bipolar:
#         filename = filename + "_bipolar"
#         title = title + " bipolar"
#     else:
#         filename = filename + "_unipolar"
#         title = title + " unipolar"
#         
#     if adaline:
#         filename = filename + "_adaline"
#         title = title + " adaline"
#     else:
#         filename = filename + "_perceptron"
#         title = title + " perceptron"
#         
#     plt.suptitle(title)
#     plt.savefig(filename + ".png")
#     #plt.show()
#     plt.close()
    return t
    
def _analyze(alpha, a, eps, bipolar=False, adaline=False):
    return numpy.average([float(_analyze_once(alpha, a, eps,bipolar, adaline)) for i in range(0, 10)])    

def analyze(alphas,aa,eps):
    results = [
               [[_analyze(alpha, a, eps, False, False) for alpha in alphas] for a in aa],
               [[_analyze(alpha, a, eps, False, True) for alpha in alphas] for a in aa],
               [[_analyze(alpha, a, eps, True, False) for alpha in alphas] for a in aa],
               [[_analyze(alpha, a, eps, True, True) for alpha in alphas] for a in aa]
            ]
#    alpha
    for i in range(0, len(aa)):
        a = aa[i]
        filename = title = "range = (" + str(-a) + "," + str(a) + ")"
        plt.suptitle(title)
        plt.plot(alphas, [results[0][i][j] for j in range(0, len(alphas))], "r", label="unipolar, perceptron")
        plt.plot(alphas, [results[1][i][j] for j in range(0, len(alphas))], "g", label="unipolar, adaline")
        plt.plot(alphas, [results[2][i][j] for j in range(0, len(alphas))], "b", label="bipolar, perceptron")
        plt.plot(alphas, [results[3][i][j] for j in range(0, len(alphas))], "y", label="bipolar, adaline")
        plt.legend(loc='upper left')
        plt.axis([alphas[0],alphas[len(alphas)-1], -1, 25])
        plt.xlabel("alpha")
        plt.ylabel("time")
        plt.savefig(filename + ".png")
        plt.close()
        
        
#    from to
    for i in range(0, len(alphas)):
        filename = title = "alpha=" + str(alphas[i])
        plt.suptitle(title)
        plt.plot(aa, [results[0][j][i] for j in range(0, len(aa))], "r", label="unipolar, perceptron")
        plt.plot(aa, [results[1][j][i] for j in range(0, len(aa))], "g", label="unipolar, adaline")  
        plt.plot(aa, [results[2][j][i] for j in range(0, len(aa))], "b", label="bipolar, perceptron")
        plt.plot(aa, [results[3][j][i] for j in range(0, len(aa))], "y", label="bipolar, adaline")
        plt.legend(loc='upper left')
        plt.axis([aa[0],aa[len(aa)-1], -1, 25])
        plt.xlabel("range")
        plt.ylabel("time")
        plt.savefig(filename + ".png")
        plt.close()

'''
analyze(
        [0.1*i for i in range(0, 5*10 + 1)], 
        [0.1*i for i in range(0, 5*10 + 1)],
        0.1
        )
'''
        
print _analyze_once(1000, 0.8, 0.1, False, False, 700)