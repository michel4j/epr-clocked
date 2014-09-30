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
import sys
import time
import gzip
import os

# CONSTANTS
EMIS_TIME_MAX = 0.07 # maximum Time between emission events us
EMIS_TIME_MIN = 0.014 # minimum Time between emission events us
EMIS_ALPHA = 1.05      # beta = 6 - alpha, for beta distribution of time differences
TIME_SCALE = numpy.pi*3e-2 # some constant related to the frequency of the particles

class Source(object):
    """Generate and emit two particles with hidden variables"""
    def __init__(self, spin=0.5):
        self.n = 2*spin
        self.phase = numpy.pi*self.n
        self.left = []
        self.right = []
        self.time = 0.0  # Synchronized clock in millisecond units
        self.ps = -1 + 1/numpy.sqrt(0.25+ numpy.linspace(0, 0.75, 1000))
        
    def emit(self):
        e = numpy.random.uniform(0.0, 2*numpy.pi)
        self.time += EMIS_TIME_MIN + numpy.random.beta(EMIS_ALPHA, 6 - EMIS_ALPHA)*EMIS_TIME_MAX
        
        # reality check, randomly eliminate 0.1% of particles, ~0.05 on each side. 
        # (ie >99.9% paired particles)
        # > 4 sigma on either tail of gaussian
        rc = numpy.random.normal(loc=0.0, scale=1.0)
        p = numpy.random.choice(self.ps)
        if rc < 4.0: 
            self.left.append((self.time, e, self.n, TIME_SCALE, p)) # add Left particle            
        if rc > -4.0:
            self.right.append((self.time, e+self.phase,  self.n, TIME_SCALE, p)) # add Right particle 
        # kind of ideal emission time but to be more realistic, we can add some 
        # jitter to each particle's time

    def save(self, fname, a):
        f = gzip.open(fname, 'wb')
        numpy.save(f, a)
        f.close()

    def run(self, duration=60.0):
        start_t = time.time()
        print "Generating spin-%0.1f particle pairs" % (self.n/2.0)
        while time.time() - start_t <= duration:
            self.emit()
            sys.stdout.write("\rTime left: %8ds" % (duration - time.time() + start_t))
            sys.stdout.flush()
        self.save('SrcLeft.npy.gz', numpy.array(self.left))
        self.save('SrcRight.npy.gz', numpy.array(self.right))
        print
        print "%d particles in 'SrcLeft.npy.gz'" % (len(self.left))
        print "%d particles in 'SrcRigh.npy.gz'" % (len(self.right))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: \n\t source.py <spin> <duration in seconds>\n"
    else:
        if os.path.exists('SEED'):
            numpy.random.seed(numpy.loadtxt('SEED')[0])
        spin, duration = map(float, sys.argv[1:])
        source = Source(spin=spin)
        source.run(duration=duration)
