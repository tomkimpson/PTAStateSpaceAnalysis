from enterprise.pulsar import Pulsar as enterprise_pulsar
import glob

datadir = ''
parfiles = sorted(glob.glob(datadir + '*.par'))
timfiles = sorted(glob.glob(datadir + '*.tim'))


psrs = []
for p, t in zip(parfiles, timfiles):
    psr = enterprise_pulsar(p, t)
    psrs.append(psr)



print(psrs)
