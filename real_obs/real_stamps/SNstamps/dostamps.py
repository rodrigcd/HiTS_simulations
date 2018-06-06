import os, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.io.fits as fits

# data dir
HiTSDIR = "/home/apps/astro/data"

# read SN locations
df = pd.read_csv("SNHiTS.txt", sep = "\s+")

# stamp size
nstamp = 21
nstamph = int(nstamp/2)

# lopp among SN
for idx, SN in df.iterrows():

    # skip if npy files exist
    if os.path.exists("npy/%s_MJDs.npy" % SN.name) and os.path.exists("npy/%s_stamps.npy" % SN.name):
        continue

    # year
    yr = SN.SNname[6:8]
    if yr == "14":
        continue

    #if SN.SNname != "SNHiTS15L":
    #    continue

    # print SN name
    print(SN.SNname)

    # central pixel
    ipix = SN.i - 1
    jpix = SN.j - 1

    # reference epoch
    refepoch = 2

    # filenames
    field = "Blind%sA_%02i" % (yr, SN.field)
    datadir = "%s/DATA/%s/%s" % (HiTSDIR, field, SN.CCD)
    refname = "%s_%s_%02i_image_crblaster.fits" % (field, SN.CCD, refepoch)

    # list directory
    files = os.listdir(datadir)

    # stamps and MJDs array
    stamps = []
    MJDs = []
    
    # compile pattern
    pattern = re.compile("%s_%s_(.*?)_image_crblaster_grid%02i_lanczos2.fits" % (field, SN.CCD, refepoch))
    
    # loop among files
    for f in sorted(files):

        # if file is reference of matches pattern
        if f == refname or pattern.match(f):
            print(f)
            
            # read data
            data = fits.open("%s/%s" % (datadir, f))
            header = data[0].header
            
            # if filters is g
            if header["FILTER"][0] == "g":

                # recover stamp
                stamp = data[0].data[ipix - nstamph: ipix + nstamph + 1, jpix - nstamph: jpix + nstamph + 1]
                # append stamp and MJD
                stamps.append(stamp)
                MJDs.append(header["MJD-OBS"])
    
    # save stamps and MJDs
    MJDs = np.array(MJDs)
    stamps = np.array(stamps)
    np.save("npy/%s_MJDs.npy" % SN.SNname, MJDs) 
    np.save("npy/%s_stamps.npy" % SN.SNname, stamps) 

    # plot stamps
    doplot = True
    if not doplot:
        continue
    fig, ax = plt.subplots(ncols = len(stamps), figsize = (len(stamps), 1.1))
    for idxstamp, stamp in enumerate(stamps):
        
        ax[idxstamp].imshow(stamp, interpolation = "nearest")
        ax[idxstamp].set_title(MJDs[idxstamp], fontsize = 5)
        ax[idxstamp].axis('off')
        ax[idxstamp].axhline(nstamph, lw = 1, c = 'k', alpha = 0.5)
        ax[idxstamp].axvline(nstamph, lw = 1, c = 'k', alpha = 0.5)

    fig.subplots_adjust(wspace = 0.02, hspace = 0.02)
    plt.savefig("plots/%s_%s_%s_%04i-%04i.png" % (SN.SNname, field, SN.CCD, ipix, jpix), bbox_inches = 'tight')

    
