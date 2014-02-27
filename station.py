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
import sys
import numpy
import gzip
from scipy import stats


class Station(object):
    """Detect a particle with a given/random setting"""
    def __init__(self, name, particles):
        self.name = name
        self.particles = particles
        self.results = numpy.empty((len(particles), 3))
        self.ps = 0.5*numpy.sin(numpy.linspace(0, numpy.pi/6, 10000))**2
        
    def detect(self, particle, setting, i):
        """Calculate the station outcome for the given `particle`"""
        a = setting 
        te, e, n, ts = particle
        C =  (0.5/numpy.pi)*((-1)**n)*numpy.cos(n*(e - a))
        p = numpy.random.choice(self.ps)
        
        # time it takes to rotate particle vector to channel vector.
        td = abs(C) < p and ts*(p-abs(C)) or 0.0 
        self.results[i] = numpy.array([te + td, a, numpy.sign(C)])

    def save(self, fname):
        """Save the results"""
        f = gzip.open(fname, 'wb')
        numpy.save(f, self.results)
        f.close()

    def run(self, angles):
        print "Detecting particles for %s's arm" % self.name
        for i, particle in enumerate(self.particles):
            setting = numpy.random.choice(angles)
            sys.stdout.write("\rParticles detected: %8d" % (i+1))
            self.detect(particle, setting, i)
        print "\nDone!"
        self.save("%s.npy.gz" % self.name)
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: \n\t station.py <ArmSrcFile> <StationName> seting1,setting2,setting3,...\n"
    else:
        if len(sys.argv) == 3: # no angles, use steps of 2.5 from 0 to 360
            angles = numpy.linspace(0, numpy.pi*2, 33)
        else:
            angles = numpy.radians(numpy.array(map(float, sys.argv[3].split(','))))
        particles = numpy.load(gzip.open(sys.argv[1], 'rb'))
        station = Station(sys.argv[2], particles)
        station.run(angles)
        
