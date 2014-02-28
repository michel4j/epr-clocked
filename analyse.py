#!/usr/bin/env python

#    Clocked EPR simulation violating the CHSH
#    Copyright (C) 2014  Michel Fodje
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from __future__ import division
import numpy
from scipy import stats
import random
import time
import sys
import matplotlib
import pprint
import gzip

matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt
from matplotlib import rcParams, colors
import matplotlib.gridspec as gridspec

rcParams['legend.loc'] = 'best'
rcParams['legend.fontsize'] = 8.5
rcParams['legend.isaxes'] = False
rcParams['figure.facecolor'] = 'white'
rcParams['figure.edgecolor'] = 'white'


PARTICLE_SPIN = 0.5 # overwritten by command line parameter
ANGLE_RESOLUTION = 0.25
COINC_WINDOW = 4.0e-4

   
def analyse(st1="Alice", st2="Bob"):
    """Perform analysis on saved output files after the simulation is done"""
    alice_raw = numpy.load(gzip.open('%s.npy.gz' % st1)) # angle, outcome 
    bob_raw = numpy.load(gzip.open('%s.npy.gz' % st2))  # angle, outcome 
    
    
    print "No. of detected particles, non-zero outcomes only"
    print "\tAlice: %15d\n\t  Bob: %15d\n" % (len(alice_raw[alice_raw[:,-1] != 0.0]), 
                                      len(bob_raw[bob_raw[:,-1] != 0.0]))

    # find matching pairs using standard algorithm 
    ai, bi = find_coincs(alice_raw[:,0], bob_raw[:,0])
    alice_orig = alice_raw[ai,:]
    bob_orig = bob_raw[bi,:]

    
    # total time is greater than random number less than half the range
    abt_orig = numpy.abs(alice_orig[:,0] - bob_orig[:,0])
    sl_co = (abt_orig <= COINC_WINDOW)
        
    alice = alice_orig[sl_co,:]
    bob = bob_orig[sl_co,:]
    
    ab_orig = numpy.abs(alice_orig[:,-2] - bob_orig[:,-2])
    ab_orig[ab_orig < 0] += 2*numpy.pi
    abdeg_orig = val(numpy.degrees(ab_orig))   
    adeg_orig  = val(numpy.degrees(alice_orig[:,-2]))
    bdeg_orig  = val(numpy.degrees(bob_orig[:,-2]))

    ab = numpy.abs(alice[:,-2] - bob[:,-2])
    ab[ab < 0] += 2*numpy.pi
    abdeg = val(numpy.degrees(ab))
    adeg  = val(numpy.degrees(alice[:,-2]))
    bdeg  = val(numpy.degrees(bob[:,-2]))
             
    # Find all settings used in simulation
    angles = numpy.degrees(numpy.unique(numpy.concatenate((alice_raw[:,-2], bob_raw[:,-2]))))
    x = angles
    Eab = numpy.zeros_like(x)
    Nab = numpy.zeros_like(x)
    ceff = numpy.zeros_like(x)

    for i, a in enumerate(x):
        sel = (numpy.abs(abdeg - a) < ANGLE_RESOLUTION)
        sel_orig = (numpy.abs(abdeg_orig - a) < ANGLE_RESOLUTION)
        Nab[i] = sel.sum()
        Eab[i] = Nab[i] > 0.0  and (alice[sel,-1]*bob[sel,-1]).mean() or 0.0
        ceff[i] = (sel.sum()/sel_orig.sum())
            
    # Display results   
    a = val(0.0)
    ap = val(45.0)
    b = val(22.5)
    bp = val(67.5)
        
    DESIG = {0 : "0, 22.5", 1: "0, 67.5", 2: "45, 22.5", 3: "45, 67.5"}    
    CHSH = []
    QM = []
    
    print "\nCalculation of expectation values"
    print "%10s %10s %10s %10s %10s %10s" % (
            'Settings', 'N_ab', 'Trials', '<AB>_sim', '<AB>_qm', 'StdErr_sim')
    for k,(i,j) in enumerate([(a,b),(a,bp), (ap,b), (ap, bp)]):
        As = (adeg==i)
        Bs = (bdeg==j)
        Ts = (As & Bs)
        OTs = ((adeg_orig==i) & (bdeg_orig==j))
        Ai = alice[Ts, -1] 
        Bj = bob[Ts, -1]
        Cab_sim = (Ai*Bj).mean()
        Cab_qm = QMFunc(numpy.radians(j-i), PARTICLE_SPIN)
        
        print "%10s %10d %10d %10.3f %10.3f %10.3f" % (DESIG[k], Ts.sum(), 
                    OTs.sum(), Cab_sim, Cab_qm, numpy.abs(Cab_sim/numpy.sqrt(Ts.sum())))
        CHSH.append(Cab_sim)
        QM.append(Cab_qm )
    
    sel_same = (abdeg == 0.0)
    sel_oppo = (abdeg == 90.0)
    SIM_SAME = (alice[sel_same,-1]*bob[sel_same,-1]).mean()
    SIM_DIFF = (alice[sel_oppo,-1]*bob[sel_oppo,-1]).mean()
       
    print
    print "\tSame Angle <AB> = %+0.2f" % (SIM_SAME)
    print "\tOppo Angle <AB> = %+0.2f" % (SIM_DIFF)
    print "\tCHSH: <= 2.0, Sim: %0.3f, QM: %0.3f" % (abs(CHSH[0]-CHSH[1]+CHSH[2]+CHSH[3]), abs(QM[0]-QM[1]+QM[2]+QM[3]))
    #print "\tCoincidence Efficiency:  %0.1f %%" % (100.0*ceff.mean())  

              
    gs = gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs[:,:])    
    ax1.plot(x, Eab, 'm-x', label='Model: E(a,b)', lw=0.5)
    ax1.plot(x, QMFunc(numpy.radians(x), PARTICLE_SPIN), 'b-+', label='QM', lw=0.5)
    bx, by = BellFunc(PARTICLE_SPIN)
    ax1.plot(bx, by, 'r--')
    ax1.legend()
    ax1.set_xlim(0, 360)
        
    #ax2 =  plt.subplot(gs[1,:])       
    #ax2.plot(x, 100*ceff, 'b--', label='% Coincidence Efficiency')
    #ax2.set_ylim(0,105)
    #ax2.legend()    
    #ax2.set_xlim(0, 360)
    
    plt.savefig('analysis-spin-%g.png' % PARTICLE_SPIN, dpi=72)
    
    print "\nStatistics of residuals between exact QM curve and Simulation"
    sts = dict(zip(['Length', 'Range', 'Mean', 'Variance', 'Skew', 'Kurtosis'], stats.describe((Eab - QMFunc(numpy.radians(x), PARTICLE_SPIN)))))
    for k, v in sts.items():
        if isinstance(v, tuple):
            vf = ' : '.join(['%0.4g' % vi for vi in v])
        else:
            vf = '%0.4g' % v
        print '%10s: %15s' % (k, vf)

    plt.show()

def QMFunc(a, spin=PARTICLE_SPIN):
    if spin == 0.5:
        return -numpy.cos(a)
    else:
        return numpy.cos(2*a)

def BellFunc(spin=PARTICLE_SPIN):
    if spin == 0.5:
        return [0.0, 180.0, 360.0], [-1.0, 1.0, -1.0]   
    else:
        return [0.0, 90.0, 180.0, 270.0, 360.0], [1.0, -1.0, 1.0, -1.0,  1.0]
    
def val(x):
    return numpy.round(x/ANGLE_RESOLUTION)*ANGLE_RESOLUTION
        
def find_coincs(a_times, b_times):
    """
    Coincidence detection algorithm adapted from Jan Ake Larsson's version
    in BellTiming.
    
    http://people.isy.liu.se/jalar/belltiming/
    <jalar@mai.liu.se>
    This function is copyrighted to Jan Ake Larsson and Licensed under the GPL
    See the License file at the above website for details on re-use
    """
    np = numpy
    # Index is the index of Alice's data, everywhere but in b_times.
    #
    # Find the immediately following detections at Bob's site.
    # Only do \.../
    # If needed I'll add /... and ...\ starts and ends later.
    # An entry == len(b_times) here means no detection follows
    following=np.searchsorted(b_times,a_times)
    # Uniqify: At most one makes a pair with another
    i=np.nonzero(following[:-1]!=following[1:])[0]
    ai_f= i
    bi_f=following[ai_f]
    ai_p=i+1
    bi_p=following[ai_p]-1
    # At this point, bi_f contains Bob's index of the detection
    # following the ones that have the indices ai_f at Alice.
    #
    # Time difference calculation
    diff_f=b_times[bi_f]-a_times[ai_f]
    diff_p=a_times[ai_p]-b_times[bi_p]
    # Link notation below: \ a detection at Bob follows a detection at
    # Alice, / a detection at Bob precedes a detection at Alice.
    # Detect chains: a_chain /\ b_chain \/ 
    a_chain = ai_f[1:]==ai_p[:-1]
    b_chain = bi_p==bi_f
    a_f_smaller_p = diff_f[1:]<diff_p[:-1]
    b_p_smaller_f = diff_p<diff_f
    while len(np.nonzero(a_chain)[0]) or len(np.nonzero(b_chain)[0]):
        #print ".",
        # Chain /\/
        # If such a chain is found and the middle time is less
        # than the outer times, remove /-/
        #print_moj("  ","/\/ ", a_chain[:30]*b_chain[1:31])
        #print_moj("  ","/\/ ", a_chain[:30]*a_f_smaller_p[:30]*b_chain[1:31])
        #print_moj("  ","/\/ ", a_chain[:30]*a_f_smaller_p[:30]*b_chain[1:31]*(1-b_p_smaller_f[1:31]))
        i=np.nonzero(a_chain*a_f_smaller_p*b_chain[1:]*(1-b_p_smaller_f[1:]))[0]
        ai_p[i] = -1
        bi_p[i] = -1
        ai_p[i+1] = -1
        bi_p[i+1] = -1
        # Chain \/\
        # If such a chain is found and the middle time is less
        # than the outer times, remove \-\
        #print_moj("","\/\ ", a_chain[:30]*b_chain[:30])
        #print_moj("","\/\ ", a_chain[:30]*(1-a_f_smaller_p[:30])*b_chain[:30])
        #print_moj("","\/\ ", a_chain[:30]*(1-a_f_smaller_p[:30])*b_chain[:30]*b_p_smaller_f[:30])
        i=np.nonzero(a_chain*(1-a_f_smaller_p)*b_chain[:-1]*b_p_smaller_f[:-1])[0]
        ai_f[i] = -2
        bi_f[i] = -2
        ai_f[i+1] = -2
        bi_f[i+1] = -2
        # Chain /\-
        # If such a chain is found and the ending time is less
        # than the previous time, remove /--
        #print_moj("  ","/\- ", a_chain[:30]*(1-b_chain[1:31]))
        #print_moj("  ","/\- ", a_chain[:30]*a_f_smaller_p[:30]*(1-b_chain[1:31]))
        i=np.nonzero(a_chain*a_f_smaller_p*(1-b_chain[1:]))[0]
        ai_p[i] = -1
        bi_p[i] = -1
        # Chain \/-
        # If such a chain is found and the ending time is less
        # than the previous time, remove \--
        #print_moj("","\/- ", (1-a_chain[:30])*b_chain[:30])
        #print_moj("","\/- ", (1-a_chain[:30])*b_chain[:30]*b_p_smaller_f[:30])
        i=np.nonzero((1-a_chain)*b_chain[:-1]*b_p_smaller_f[:-1])[0]
        ai_f[i] = -2
        bi_f[i] = -2
        # Chain -\/
        # If such a chain is found and the starting time is less
        # than the following time, remove --/
        #print_moj("  ","-\/ ", (1-a_chain[:30])*b_chain[1:31])
        #print_moj("  ","-\/ ", (1-a_chain[:30])*b_chain[1:31]*(1-b_p_smaller_f[1:31]))
        i=np.nonzero((1-a_chain)*b_chain[1:]*(1-b_p_smaller_f[1:]))[0]
        ai_p[i+1] = -1
        bi_p[i+1] = -1
        # Chain -/\
        # If such a chain is found and the middle time is less
        # than the following time, remove --\
        #print_moj("","-/\ ", a_chain[:30]*(1-b_chain[:30]))
        #print_moj("","-/\ ", a_chain[:30]*(1-a_f_smaller_p[:30])*(1-b_chain[:30]))
        i=np.nonzero(a_chain*(1-a_f_smaller_p)*(1-b_chain[:-1]))[0]
        ai_f[i+1] = -2
        bi_f[i+1] = -2
        a_chain = ai_p[:-1]==ai_f[1:]
        b_chain = bi_p==bi_f
        #print "a_chain", a_chain
        #print "b_chain", b_chain
    return ai_f[ai_f>0], bi_f[bi_f>0]
             
if __name__ == '__main__':
    if len(sys.argv) == 2:
        PARTICLE_SPIN = float(sys.argv[1])
        analyse()
    else:
        print "Usage: \n\t analyse.py <spin>\n"
        
