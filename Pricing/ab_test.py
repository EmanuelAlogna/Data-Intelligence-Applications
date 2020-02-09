import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

class SequentialABTest():
    def __init__(self, p1,p2,alpha=0.05,beta=0.8):
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha
        self.beta = beta

    def collect_samples(self,n_samples):
        x1 = [np.random.binomial(1,self.p1) for _ in range(0,int(n_samples/2))]
        x2 = [np.random.binomial(1,self.p2) for _ in range(0,int(n_samples/2))]
        return x1,x2

    def test(self,x1,x2,price1,price2):
        mu1 = np.mean(x1) * price1
        mu2 = np.mean(x2) * price2
        n1 = np.array(x1).shape[1]
        n2 = np.array(x2).shape[1]
        var1 = self.p1 * (1-self.p1)
        var2 = self.p1 * (1-self.p1)
        z = (mu1 - mu2) / np.sqrt(((var1/n1)+(var2/n2)))
        print('Z: {}'.format(z))
        p_val = 1 - st.norm.cdf(z)
        return p_val

    def best_candidate(self,x1,x2,price1,price2):
        p_val = self.test(x1,x2,price1,price2)
        print('p_val = {}'.format(p_val))
        print('alpha = {}'.format(self.alpha))

        if(p_val < self.alpha):
            print('Accetto ipotesi alternativa')
            return 1
        else:
            print('Accetto ipotesi nulla')
            return 0

    def calculate_sample_size(self):
        z_alpha = abs(st.norm.ppf(self.alpha/2))
        z_beta = abs(st.norm.ppf(self.beta))
        var = self.p1 * (1 - self.p1) + self.p2 * (1 - self.p2)
        min_var = abs(self.p1 - self.p2)
        n_samples = (((z_alpha + z_beta)**2) * var) / (min_var**2)
        return int(n_samples)

    def calculate_sample_size2(self):
        z_alpha = st.norm.ppf(1-self.alpha)
        z_beta = abs(st.norm.ppf(self.beta))
        var = self.p1 * (1 - self.p1) + self.p2 * (1 - self.p2)
        min_var = abs(self.p1 - self.p2)
        n_samples = (((z_alpha + z_beta)**2) * var) / (min_var**2)
        return int(n_samples)

np.random.seed(12)
p = [0.73, 0.61, 0.5, 0.4]
prices = {0.73:6.5, 0.61:8, 0.5: 9.5,0.4: 10.5}
p.reverse()
reward = []
control = p.pop()
n_experiments = 100
best_price = p[0]

plt.figure(0)

while p:
    rew_control = []
    rew_test = []
    test = p.pop()
    print('Testing {} vs {}'.format(control,test))
    ab_tester = SequentialABTest(p1=control, p2=test,alpha=0.05)
    n_samples = ab_tester.calculate_sample_size2()
    print('N samples: {}'.format(n_samples))

    x1, x2 = ab_tester.collect_samples(n_samples)

    rew_control.append(x1)
    rew_test.append(x2)
    winner = ab_tester.best_candidate(rew_control, rew_test,prices[control],prices[test])
    mean_control = np.mean(rew_test)

    mean_test = np.mean(rew_test)
    reward += n_samples * [(mean_control * prices[control] + mean_test * prices[test]) / 2]
    plt.plot(reward, 'r')
    if(winner == 1):
        print('Best price: {}'.format(control))
        best_price = control
    else:
        print('Best price: {}'.format(test))
        control = test
        best_price = test

    #reward += n_samples * [(control*prices[control] + test*prices[test])/2]
    plt.plot(reward,'r')

a = [np.random.binomial(1,best_price) for _ in range(0,300)]
reward += len(a) * [np.mean(a)* prices[best_price]]
clairvoyant = 800* [best_price * prices[best_price]]
plt.plot(reward, 'r',label='AVG Reward')
plt.plot(clairvoyant, 'b--',label='Clairvoyant')
plt.legend()
plt.show()

