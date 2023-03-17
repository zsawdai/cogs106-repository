import scipy as spi
import matplotlib.pyplot as plt
import numpy as np

class SignalDetection:
    
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
    
    def H(self):
        return (self.hits / (self.hits + self.misses))

    def FA(self):
        return (self.falseAlarms / (self.falseAlarms + self.correctRejections))

    def d_prime(self):
        return (spi.stats.norm.ppf(self.H()) - spi.stats.norm.ppf(self.FA()))

    def criterion(self):
        return ((-0.5) * (spi.stats.norm.ppf(self.H()) + spi.stats.norm.ppf(self.FA())))
    
    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)
    
    def __mul__(self, scalar):
        return SignalDetection(self.hits * scalar, self.misses * scalar, self.falseAlarms * scalar, self.correctRejections * scalar)

    @staticmethod
    def plot_roc(sdtList):
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic Curve")
        if isinstance(sdtList, list):
            for i in range(len(sdtList)):
                s = sdtList[i]
                plt.plot(s.FA(), s.H(), 'o', color = 'black', markersize = 10)
        else:
            plt.plot(sdtList.FA(), sdtList.H(), 'o', color = 'black', markersize = 10)
        x, y = np.linspace(0,1,100), np.linspace(0,1,100)
        plt.plot(x,y, '--', color = 'black')
        plt.grid()

    def plot_sdt(self):
        noise_x = np.arange(-4, 4, 0.1)
        noise_y = spi.stats.norm.pdf(noise_x, 0, 1)
        signal_x = np.arange(-4, 4, 0.1)
        signal_y = spi.stats.norm.pdf(noise_x, self.d_prime(), 1)
        plt.plot(noise_x, noise_y, label = "Noise", color = 'blue')
        plt.plot(signal_x, signal_y, label = "Signal", color = 'green')
        plt.axvline(x = ((self.d_prime() / 2) + self.criterion()), label = "k", color = 'r', linestyle = '--')
        x_distance = [0, self.d_prime()]
        y_distance = [0.4, 0.4]
        plt.plot(x_distance, y_distance, '--', label = "Distance", color = 'black')
        plt.plot(0,0, 'o', label = '0', color = 'blue')
        plt.plot(self.d_prime(), 0, 'o', label = 'D\'', color = 'green')
        plt.title("Signal Detection Theory Curve")
        plt.xlabel("Response")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
      sdtList = []
      for i in range(len(criteriaList)):
          k = criteriaList[i] + (dprime/2)

          hits, falseAlarms = np.random.binomial(n=[signalCount, noiseCount], p=[1 - spi.stats.norm.cdf(k - dprime),1 - spi.stats.norm.cdf(k)])
          misses, correctRejections = signalCount - hits, noiseCount - falseAlarms

          sdtList.append(SignalDetection(hits, misses, falseAlarms, correctRejections))
      return sdtList

    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        return -((self.hits * np.log(hit_rate)) + (self.misses * np.log(1-hit_rate)) + (self.falseAlarms * np.log(false_alarm_rate)) + (self.correctRejections * np.log(1-false_alarm_rate)))

    @staticmethod
    def rocCurve(falseAlarmRate, a):
        return spi.stats.norm.cdf(a + spi.stats.norm.ppf((falseAlarmRate)))
    
    @staticmethod
    def fit_roc(sdtList):
        SignalDetection.plot_roc(sdtList)
        a = 0
        minimize = spi.optimize.minimize(fun = SignalDetection.rocLoss, x0 = a, method = 'nelder-mead', args = (sdtList))
        losscurve = []
        for i in range(0,100,1):
          losscurve.append((SignalDetection.rocCurve(i/100, float(minimize.x))))
        plt.plot(np.linspace(0,1,100), losscurve, '-', color='red')
        aHat = minimize.x
        return float(aHat)

    @staticmethod
    def rocLoss(a, sdtList):
        L = []
        for i in range(len(sdtList)):
            s = sdtList[i]
            predicted_hit_rate = s.rocCurve(s.FA(), a)
            L.append(s.nLogLikelihood(predicted_hit_rate, s.FA()))
        return sum(L)
