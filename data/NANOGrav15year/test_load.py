from enterprise.pulsar import Pulsar as enterprise_pulsar
import glob


parfiles = sorted(glob.glob('wideband/par/*.par'))
timfiles = sorted(glob.glob('wideband/tim/*.tim'))

psrs = []
for p, t in zip(parfiles, timfiles):
    print("Loading file {0}".format(p))
    psr = enterprise_pulsar(p, t)
    psrs.append(psr)



print(psrs)
