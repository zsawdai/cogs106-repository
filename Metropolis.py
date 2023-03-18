import numpy as np
import scipy as spi

class Metropolis:

    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.initialState = initialState
        self.samples = []
        self.sDevK = None
        
    def __accept(self, proposal, current):
        acceptanceProb = min(0, (self.logTarget(proposal) - self.logTarget(current)))
        u = (np.random.uniform(0,1))
        if np.exp(acceptanceProb) > u:
            return True
        else:
            return False

    def adapt(self, blockLengths):
        sDevCurr, targetRate, accepted, notAccepted = 1, 0.4, 0, 0
        theta = self.initialState
        for i in range(len(blockLengths)):
            proposal = np.random.uniform(theta,sDevCurr)
            accept = self.__accept(proposal=proposal, current=theta)
            if accept == True:
                accepted = accepted+1
                theta = proposal
            else:
                notAccepted = notAccepted+1
            acceptanceRate = accepted / (accepted+notAccepted)
            sDevNew = sDevCurr * ((targetRate / acceptanceRate)**1.1)
            sDevCurr = sDevNew
            self.sDevK = sDevNew
        return self

    def sample(self, nSamples):
        theta = self.initialState
        sDevK = self.sDevK
        for i in range(nSamples):
            proposed = np.random.normal(theta, sDevK)
            accept = self.__accept(proposed, theta)
            self.samples.append(theta)
            if accept == True:
                theta = proposed
        return self
    
    def summary(self):
        self.samples = self.samples[100:]
        mean, stdDev = np.mean(self.samples), np.std(self.samples, ddof=1)
        posteriorMean, posteriorStdDev = (0 / 1**2 + mean / stdDev**2) / (1 / 1**2 + 1 / stdDev**2), np.sqrt(1 / (1 / 1**2 + 1 / stdDev**2))
        c025, c975 = spi.stats.norm.ppf(0.025, loc = posteriorMean, scale = posteriorStdDev), spi.stats.norm.ppf(0.975, loc = posteriorMean, scale = posteriorStdDev)
        return {'mean': mean,'c025': c025,'c975': c975}
