#!/usr/bin/env python3
# -*- coding: utf-8; -*-

# Eppes command line routine
# Reads files defined in a conf file given as input argument with -c (default name is eppes.cfg).
# Overwrites mufile, sigfile, wfile and nfile on output.

import numpy as np
import eppes
# from eppes import config

def main(*args):

    # Read configuration
    opts = eppes.config.parseargs(*args[1:])

    # define bounds
    if len(opts.boundsfile)>0:
        bounds = np.loadtxt(opts.boundsfile)
    else:
        bounds = None

    # load mu and sig from file
    mu = np.loadtxt(opts.mufile)
    sig = np.loadtxt(opts.sigfile)

    if (mu.shape[0] != sig.shape[0]) | (mu.shape[0] != sig.shape[1]) :
        print('mu and sig shapes do not match')
        return 1

    # sample only, no update
    if opts.sampleonly == 1:
        newsample = eppes.propose(opts.nsample,mu,sig,bounds=bounds)
        if opts.lognor:
            newsample = np.exp(newsample)
        if opts.verbosity:
            print(mu)
            print(sig)
            print(newsample)
        np.savetxt(opts.sampleout,newsample)
        return 0

    # load W and N, current sample and the scores
    w = np.loadtxt(opts.wfile)
    n = np.loadtxt(opts.nfile)
    oldsample = np.loadtxt(opts.samplein)
    scores = np.loadtxt(opts.scorefile)
    nsample = oldsample.shape[0]
    if opts.lognor:
        oldsample = np.log(oldsample)


    eppes.tomatrix(scores)
    if scores.shape[0] != nsample:
        print('scores and sample do not match')
        return 1
    # turn scores into ranks
    if opts.useranks:
        scores = eppes.rankscores(scores)
        if opts.verbosity:
            print(scores)

    if np.size(np.shape(scores)) > 1:
        if np.shape(scores)[1] > 1:
            scores = eppes.combinescores(scores,opts.combine_method)
            eppes.tomatrix(scores)
            if opts.verbosity:
                print(scores)
        
    # resample and update parameters mu,w,sig,n
    theta,wout = eppes.logresample(oldsample,scores)
    mu,w,sig,n = eppes.eppesupdate(theta,mu,w,sig,n, maxsteprel=opts.maxsteprel)

    # optionally, fix N and W
    if opts.maxn>0:
        n = min(n,opts.maxn)
    if len(opts.w00file)>0:
        w00 = np.loadtxt(opts.w00file)
        if w.shape != sig.shape:
            print('error in w00 shape, not using it')
        else:
            w = w + w00

    # generate new sample, based on updated mu and sig
    newsample = eppes.propose(opts.nsample,mu,sig,bounds=bounds)

    if opts.lognor:
        newsample = np.exp(newsample)
    
    if opts.verbosity:
        print(mu)
        print(sig)
        print(newsample)
        print(n)

    # save all back to files (they are over written!)
    np.savetxt(opts.sampleout,newsample)
    np.savetxt(opts.mufile,mu)
    np.savetxt(opts.sigfile,sig)
    np.savetxt(opts.wfile,w)
    np.savetxt(opts.nfile,np.array([n]))
    if len(opts.winfofile)>0:
        np.savetxt(opts.winfofile,wout)
    return 0

   
if __name__ == "__main__":
    import sys 
    status = main(*sys.argv)
    # need some nice way to pass failure to the caller
    if status:
        sys.exit("Some error happened during the call, status="+str(status))
