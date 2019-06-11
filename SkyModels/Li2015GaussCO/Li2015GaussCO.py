import numpy as np
from matplotlib import pyplot
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simps
import hmf
from astropy import wcs
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.fftpack as sfft
from pathlib import Path
from scipy.ndimage import gaussian_filter

#Constants
pc = 3.086e16
H0 = 100.*1e3 * 1e-6  / pc  #1/s
c  = 299792458.
k = 1.38e-23
v0 = 115e9
L0 = 3.828e26

class Li2015GaussCO:
    """
    Generate Gaussian CO cube realisations using the Li 2015 CO model.
    """

    def __init__(self, freqmin = 26, freqmax= 34, nchannels=64, Mmin = 9, Mmax=14, save=False,
                x0 = 0, y0 = 0, dtheta = 1./60., nspix = 256, ctype = ['RA---TAN', 'DEC--TAN']):
        self.SFRdata = 'sfr_release.dat'

        # Instrument
        self.freqDelta = (freqmax - freqmin)/nchannels * 1e9 # Hz
        self.freqMin   = freqmin * 1e9 # Hz
        self.freqMax   = freqmax * 1e9 # Hz
        self.nchannels = int(nchannels)
        self.freqs     = (np.arange(nchannels) + 0.5)*self.freqDelta + self.freqMin

        # Setup image
        self.crval = [x0, y0]
        self.cdelt = [dtheta, dtheta]
        self.npixs = [int(nspix), int(nspix)]
        self.crpix = [nspix/2, nspix/2]
        self.ctype = ctype
        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

        # Cosmology
        self.mycosmo   = hmf.cosmo.Cosmology()
        self.Pk   = hmf.transfer.Transfer()

        self.Mmin = Mmin
        self.Mmax = Mmax
        self.mf   = hmf.hmf.MassFunction(Mmin=self.Mmin,Mmax=self.Mmax)
        self.z    = v0/self.freqs - 1
        self.rc   = self.mycosmo.cosmo.comoving_distance(self.z).value *1e6 *pc
        dk   = self.Pk.k[1:] - self.Pk.k[:-1]
        Pmk  = self.Pk.delta_k/self.Pk.k**3 * 2 * np.pi
        Pmk  = (Pmk[1:] + Pmk[:-1])/2.
        self.inputPk = self.Pk.delta_k/self.Pk.k**3 * 2* np.pi/np.sum(Pmk*dk)
        self.pmdl = InterpolatedUnivariateSpline(self.Pk.k,self.inputPk)

        # Products
        self.cube = None

        # Initalisation
        if save:
            self.filename = 'MeanTco_freq{:.1f}-{:.1f}_N{:d}_M{:.0f}-{:.0f}.npz'.format(freqmin,
                                                                                   freqmax,
                                                                                   int(nchannels),
                                                                                   Mmin, Mmax)
        else:
            self.filename = ''
        my_file = Path(self.filename)

        # If a large job was already done, just reload that model.
        if save:
            if my_file.is_file():
                self.Tco = np.load(self.filename)['Tco']
            else:
                self.Tco = self.Halo2Tco()
                np.savez(self.filename, Tco=self.Tco)
        else:
            self.Tco = self.Halo2Tco()

    def __call__(self):
        """
        Each call to the class should generate a new cube.
        """
        # Setup power spectrum step sizes
        dt = self.cdelt[0]*np.pi/180. * np.mean(self.rc) / (1e6 *pc) # assume a mean comove dist, reasonable approximation?
        dr = np.abs(np.mean(self.rc[1:] - self.rc[:-1]))  / (1e6 *pc)
        self.step  = [dt,dt,dr]
        self.shape = [self.npixs[1],self.npixs[0],self.nchannels]
        self.cube = self.GenCube(self.step, self.shape, self.pmdl) # returns a normalised cube of haloes

        self.cube = self.cube*self.Tco[np.newaxis,np.newaxis,:]
        return self.cube

    def sphAvgPwr(self, nbins=30):
        """
        Perform 3D FFT to generate a gaussian realisation of the CO cube
        """
        x = sfft.fftfreq(self.shape[0], d=self.step[0])
        y = sfft.fftfreq(self.shape[1], d=self.step[1])
        z = sfft.fftfreq(self.shape[2], d=self.step[2])
        i = np.array(np.meshgrid(x,y,z, indexing='ij'))

        # k-space distances
        k = np.array([i[j,...]**2 for j in range(i.shape[0]) ])
        k = np.sqrt(np.sum(k, axis=0))

        kEdges = np.linspace(np.min(k),np.max(k),nbins+1)
        kMids  = (kEdges[1:] + kEdges[:-1])/2.

        Pcube = np.abs(sfft.fftn(self.cube-np.median(self.cube)))**2
        Pk = np.histogram(k.flatten(), kEdges, weights=Pcube.flatten())[0]/np.histogram(k.flatten(), kEdges)[0]

        return kMids, Pk


    def setWCS(self, crval, cdelt, crpix, ctype=['RA---TAN', 'DEC--TAN']):
        """
        Setup WCS
        """
        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.crval = crval
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.ctype = ctype

    def GenCube(self, step, shape, pmodel, random=np.random.normal):
        """
        Perform 3D FFT to generate a gaussian realisation of the CO cube
        """
        x = sfft.fftfreq(shape[0], d=step[0])
        y = sfft.fftfreq(shape[1], d=step[1])
        z = sfft.fftfreq(shape[2], d=step[2])
        i = np.array(np.meshgrid(x,y,z, indexing='ij'))

        r = np.array([i[j,...]**2 for j in range(i.shape[0]) ])
        r = np.sqrt(np.sum(r, axis=0))

        img  = pmodel(r)
        vals = random(size=r.shape)
        vals -= np.median(vals)
        #vals /= np.std(vals)
        phases = sfft.fftn(vals)
        cube = (sfft.ifftn(np.sqrt(img)*phases))
        return np.real(cube)


    def Halo2SFR(self, z, mass):
        """
        Interpolate SFR from Behroozi, Wechsler, & Conroy 2013a (http://arxiv.org/abs/1207.6105) and 2013b ((http://arxiv.org/abs/1209.3013)
        """
        zp1, logm, logsfr, _  = np.loadtxt(self.SFRdata).T

        logm   = np.unique(logm)
        zp1    = np.unique(np.log10(zp1))
        logsfr = np.reshape(logsfr, (logm.size, zp1.size))

        sfr_interp = RectBivariateSpline(logm, zp1,10**logsfr, kx=1,ky=1)

        return sfr_interp.ev(mass, np.log10(z+1))

    def SFR2IR(self, SFR):
        """
        Start formation rate to IR from Li et al...
        """
        return SFR * 1e10

    def IR2Lco(self, IR, alpha=1.37, beta=-1.74):
        """
        Lco in units of K km/s /pc^2
        """

        A = (np.log10(IR) - beta)/alpha
        return 10**A

    def Halo2Tco(self):
        """
        Derive mean co brightness temperature with redshift
        """
        # Here derive relation between mass and Lco for each redshift bin
        self.Tco  = np.zeros(self.z.size)

        norm = simps(self.mf.dndm, self.mf.m)
        Jy2K = c**2 / (2. * k * self.freqs**2)
        for i in range(self.z.size):
            SFR = self.Halo2SFR(self.z[i], np.log10(self.mf.m))
            IR  = self.SFR2IR(SFR)
            _Lco = self.IR2Lco(IR)

            self.mf.update(z=self.z[i])
            Lco = simps(_Lco**1*self.mf.dndm, self.mf.m)/norm*4.9e-5

            self.Tco[i] = Jy2K[i]* Lco*L0/(4*np.pi*(1+self.z[i])**2*self.rc[i]**2)
        return self.Tco


if __name__ == "__main__":
    test = Li2015GaussCO(save=True)
    cube = test()
    k, Pk = test.sphAvgPwr(nbins=256)

    pyplot.subplot(121, projection=test.wcs)
    pyplot.imshow(gaussian_filter(cube[:,:,0],3)*1e6)
    pyplot.grid()
    pyplot.colorbar(label=r'$\mu$K')
    pyplot.subplot(122)
    print(test.cube.size,)
    pyplot.plot(k, Pk * k**3 *1e12,color='k',zorder=1,linestyle='--')
    pyplot.yscale('log')
    pyplot.xscale('log')
    ylim = pyplot.ylim()
    pyplot.plot(test.Pk.k,test.inputPk*test.Pk.k**3 * np.mean(test.Tco)**2 * test.cube.size* 1e12, zorder=0)
    pyplot.xlim(np.min(k),np.max(k))
    pyplot.ylim(ylim)
    pyplot.grid()
    pyplot.xlabel(r'$k$')
    pyplot.ylabel(r'$P_k$ $k^3$')
    pyplot.tight_layout()
    pyplot.show()
