# data storage                                          -*- coding: utf-8; -*-

import pickle
import numpy as np

# order of the elements currently: mu, Sig, W, n
# load data:
# data = eppes.EppesData(), data.load('eppesrun.pk'),
# mu,Sig,W,n = data.get()
# or directly
# mu,Sig,W,n = eppes.EppesData().load('eppesrun.pk').get()

class EppesData(object):
    """
    Class to store eppes parameters
    order of the elements currently: mu, Sig, W, n
    Might change soon.
    Examples:
    x = eppes.EppesData()
    npar = 3
    for i in range(10):
        x.add(mu=rand(npar),Sig=randn(npar,npar),W=eye(npar),n=i+1) 
    x['mu']
    x['Sig'][5]
    x.save('eppesdata.pk')

    y = eppes.EppesData()
    y.load('eppesdata.pk')
    """

    def __init__(self, file=None):
        self.data = []
        self.__iternum = 0
        self.__npar = None
        self.__current = {'mu': None, 'Sig': None, 'W': None, 'n': None}
        self.names = []
        if file is not None:
            self.load(file)

    @property
    def npar(self):
        """Number of parameters"""
        return self.__npar

    @property
    def iternum(self):
        """Number of iterations"""
        return self.__iternum

    @property
    def value(self):
        """Current parameters as tuple"""
        return self.__current['mu'], self.__current['Sig'], self.__current['W'], self.__current['n']

    @property
    def current(self):
        """Current parameters as dictionary"""
        return self.__current

    def add(self, mu, Sig, W, n):
        """Append data to eppes internal storage"""
        self._add1(mu, Sig, W, n)
        self.__iternum += 1
        self.data.append(self.__current.copy())

    def _add1(self, mu, Sig, W, n):
        """ Makes the values current """
        mu = np.asarray(mu, dtype=np.float64).ravel()
        if self.__npar is None:
            self.__npar = mu.size
        Sig = np.asarray(Sig, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        n = np.asarray(n)

        if mu.size != self.__npar:
            raise ValueError('mu size does not match')
        if Sig.shape != (self.__npar, self.__npar):
            raise ValueError('Sig size does not match')
        if W.shape != (self.__npar, self.__npar):
            raise ValueError('W size does not match')
        if n.size != 1:
            raise ValueError('n size does not match')

        self.__current['mu'] = mu
        self.__current['Sig'] = Sig
        self.__current['W'] = W
        self.__current['n'] = n

    def save(self, file='eppesrun.pk'):
        """ save class data """
        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, file='eppesrun.pk'):
        """ load class data """
        with open(file, 'rb') as f:
            attr = pickle.load(f)
            self.__dict__.update(attr)

    def get(self, what='all'):
        """ get all the data """
        #        if np.size(self.data) < 1:
        #            raise ValueError('No data yet')
        if what == 'all':
            mu = np.squeeze(np.stack([d['mu'] for d in self.data]))
            Sig = np.stack([d['Sig'] for d in self.data])
            W = np.stack([d['W'] for d in self.data])
            n = np.squeeze(np.stack([d['n'] for d in self.data]))
            return mu, Sig, W, n
        else:
            return np.squeeze(np.stack([d[what] for d in self.data]))

    def __getitem__(self, key):
        if type(key) == str:
            return self.get(key)
        else:
            return self.data[key]

    def __repr__(self):
        """ what to print """
        return str(self.__class__) + "\nCurrent value:\n" + str(self.__current)

    def plot(self, type='ts'):
        """default plotting for eppes results"""
        import matplotlib.pyplot as plt
        mu, Sig, W, n = self.get()
        niter = mu.shape[0]
        if type == "ts":
            plt.plot(np.arange(niter), mu)
            plt.show()
        else:
            if type == "ellipse":
                pass


class DataSet(object):
    """
    Stores data and optionally saves them
    Examples:
      x = DataSet()
      x.add(randn(3,5))
      x.add(randn(3,5))
      x.add(randn(3,5))
      x[:]
      x.save('data.pk')
    
      y = DataSet('data.pk')[:]
    """

    def __init__(self, file=None):
        self.data = []
        self.__iternum = 0
        self.__current = None
        self.__shape = None
        if file is not None:
            self.load(file)

    @property
    def value(self):
        """The current data"""
        return self.__current

    @property
    def iternum(self):
        """Number of iterations"""
        return self.__iternum

    @property
    def shape(self):
        """Data shape"""
        return self.__shape

    def add(self, x):
        """Append data to the internal storage"""
        x = np.asarray(x, dtype=np.float64)
        if self.__shape is None:
            self.__shape = x.shape
        if x.shape != self.__shape:
            raise ValueError('data shape is not consistent')
        self.__current = x
        self.__iternum += 1
        self.data.append(self.__current.copy())

    def save(self, file='data.pk'):
        """save class data to file"""
        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, file='data.pk'):
        """load class data from file"""
        with open(file, 'rb') as f:
            attr = pickle.load(f)
            self.__dict__.update(attr)

    def get(self):
        """Get all the data as numpy array"""
        return np.stack(self.data)

    def __getitem__(self, key):
        return np.stack(self.data[key])

    def __repr__(self):
        """Info on the data set"""
        return str(self.__class__) + "\n" + str(self.__iternum) + " sets" + "\nCurrent:\n" + str(self.__current)
