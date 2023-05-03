import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools
import math
from functools import total_ordering
from numpy.random import default_rng

class LJParticles:
    """
    Description of Parameters:
    
        Boundary
    Lx, Ly: box dimensions
    
        Time
    dt: time step
    tmax: duration of simulation
    dispEvery: number of time steps to be calculated per frame saved
    
        Initialization
    nx, ny: number of particles in initialized rectangular lattice (numParticles = nx * ny)
    initialKE: initial kinetic energy per particle
    rng_seed: used to randomly initialize velocities
    
        Cooling
    cool: if True, system will cool. if False, it won't.
    cool_every: number of time steps to be calculated between cooling steps
    cool_startT: time before which no cooling will happen (to allow equilibration)
    cool_endT: time after which no cooling will happen (to allow equilibration)
    cool_factor: factor by which kinetic energy is multiplied during cooling steps
    """
    
    def __init__(self, nx = 2, ny = 2, Lx = 4, Ly = 4, initialKE = 0, dt = 0.01, tmax = 5, 
                 dispEvery = 10, rng_seed = 42, cool = False, cool_every=200, 
                 cool_startT=1, cool_endT=100, cool_factor=0.9):
        """
        Initializes a Lennard-Jones molecular dynamics simulation, 
        with epsilon = 1, sigma = 1 and particle mass = 1
        """
        
        self.rng = default_rng(seed = rng_seed)
        
        self.cool_every = cool_every
        self.cool_startT = cool_startT
        self.cool_endT = cool_endT
        self.cool_factor = cool_factor
        
        # Total number of particles
        self.N = nx * ny
        
        self.hitWall = False
        
        # Arrays for positions
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)
        
        # Arrays for velocities
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)
        
        # Arrays for accelerations
        self.ax = np.zeros(self.N)
        self.ay = np.zeros(self.N)
        
        # Number of particles per row and column
        self.nx = nx
        self.ny = ny
        
        # Box width and height
        self.Lx = Lx
        self.Ly = Ly
        
        # Initial kinetic energy per particle
        self.initialKE = initialKE        
        
        # time variables
        
        self.dt = dt # Time step
        self.t = 0 # Time since simulation began
        self.tmax = tmax # Time when simulation ends
        self.steps = 0 # number of simulation steps
        
        # number of advance calls per update to display
        self.dispEvery = dispEvery
        
        # Remember energies for graphs
        self.histPE = []
        self.histKE = []
        
        # Tells us if system cools over time
        self.cool = cool
        
        #set velocities and positions
        self.setVelocities()
        self.setRectangularLattice()
        
    def setVelocities(self):
        """
        Sets initial velocities according to desired kinetic energy.
        """
        vxSum = 0.0
        vySum = 0.0
        
        N = self.N
        
        # Generate random initial velocities
        for i in range(N):
            self.vx[i] += self.rng.random() - 0.5
            self.vy[i] += self.rng.random() - 0.5
            
            vxSum += self.vx[i]
            vySum += self.vy[i]
            
        # Zero the center of mass momentum
        vxcm = vxSum / N   # Center of mass velocity (numerically equal to momentum)
        vycm = vySum / N
        
        for i in range(N):
            self.vx[i] -= vxcm
            self.vy[i] -= vycm
        
        # Rescale velocities to get desired initial kinetic energy
        v2sum = 0
        
        for i in range(N):
            v2sum += self.vx[i]**2 + self.vy[i]**2
        
        kineticEnergyPerParticle = 0.5 * v2sum / N
        
        rescale = math.sqrt(self.initialKE / kineticEnergyPerParticle)
        
        for i in range(N):
            self.vx[i] *= rescale
            self.vy[i] *= rescale
        
    def applyBC(self, i):
        """
        Updates particle i according to hard-wall boundary conditions.
        Only corrects velocity, not position.
        """
        
        #commented-out lines are for different wall-hit resolution strategies.
        if self.x[i] > self.Lx:
#             self.y[i] -= (self.Lx-self.x[i]) * (self.vy[i] / self.vx[i])
#             self.x[i] = self.Lx
            self.vx[i] = -abs(self.vx[i])
            self.hitWall = True
        elif self.x[i] < 0:
#             self.y[i] += (self.x[i]) * (self.vy[i] / self.vx[i])
#             self.x[i] = 0
            self.vx[i] = abs(self.vx[i])
            self.hitWall = True
        
        if self.y[i] > self.Ly:
#             self.x[i] -= (self.Ly-self.y[i]) * (self.vx[i] / self.vy[i])
#             self.y[i] = self.Ly
            self.vy[i] = -abs(self.vy[i])
            self.hitWall = True
        elif self.y[i] < 0:
#             self.x[i] += (self.y[i]) * (self.vx[i] / self.vy[i])
#             self.y[i] = 0
            self.vy[i] = abs(self.vy[i])
            self.hitWall = True
    
    def setRectangularLattice(self):
        """
        Arranges the particles on an nx by ny rectangular lattice.
        Particles are offset from the walls by 1 unit.
        """
        
        # Horizontal and vertical spacings
        dx = (self.Lx - 2) / (self.nx - 1)
        dy = (self.Ly - 2) / (self.ny - 1)
        
        # Set initial positions
        for ix in range(self.nx):
            for iy in range(self.ny):
                i = ix + iy * self.ny
                self.x[i] = 1 + dx * ix
                self.y[i] = 1 + dy * iy
    
    def computeAcceleration(self):
        """
        Computes the acceleration of each particle due to Lennard-Jones forces.
        """
        N = self.N
        
        for i in range(N):
            self.ax[i] = 0
            self.ay[i] = 0
        
        totalPE = 0
        
        # For any two particles...
        for i in range(N):
            for j in range(i + 1, N):
                
                # Find the separations
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                
                r2 = dx**2 + dy**2
                
                oneOverR2 = 1.0 / r2
                oneOverR6 = oneOverR2**3
                
                # Calculate the Lennard-Jones force
                fOverR = 48 * oneOverR6 * (oneOverR6 - 0.5) * oneOverR2
                
                fx = fOverR * dx
                fy = fOverR * dy
                
                # Update accelerations
                self.ax[i] += fx
                self.ay[i] += fy
                self.ax[j] -= fx
                self.ay[j] -= fy
                
                # Add to PE for this run
                totalPE += 4 * (oneOverR6**2 - oneOverR6)
        return totalPE
                
    def advance(self):
        """
        Advances the simulation by one time step. Velocity is updated using the average of old and
        new acceleration (Verlet algorithm).
        """
        N = self.N
        halfdt = 0.5 * self.dt
        halfdt2 = 0.5 * self.dt**2
        
        # For each particle...
        for i in range(N):
            
            # Update position
            self.x[i] += self.vx[i] * self.dt + self.ax[i] * halfdt2
            self.y[i] += self.vy[i] * self.dt + self.ay[i] * halfdt2
            
            self.applyBC(i)
            
            # Update velocity with old acceleration
            self.vx[i] += self.ax[i] * halfdt
            self.vy[i] += self.ay[i] * halfdt
            
        totalPE = self.computeAcceleration()
        
        # Update with new acceleration
        for i in range(N):
            self.vx[i] += self.ax[i] * halfdt
            self.vy[i] += self.ay[i] * halfdt
        
        # Cool the system if desired
        if (self.cool and self.steps % self.cool_every == 0 and 
            (self.cool_startT <= self.t) and (self.t <=self.cool_endT)):
            self.cools()
        
        # Compute the system's kinetic energy
        tempKE = 0
        
        for i in range(N):
            v2 = self.vx[i]**2 + self.vy[i]**2
            tempKE += v2
        
        tempKE *= 0.5
        
        # Require total energy to be conserved by scaling kinetic energy, then 
        # compute the system's kinetic energy after correction
        
        if self.hitWall:
            velocityScaleFactor = ((self.histPE[-1] + self.histKE[-1] - totalPE) / tempKE) ** .5
            self.hitWall = False
        else:
            velocityScaleFactor = 1
        totalKE = 0
        
        for i in range(N):
            self.vx[i] *= velocityScaleFactor
            self.vy[i] *= velocityScaleFactor
            v2 = self.vx[i]**2 + self.vy[i]**2
            totalKE += v2
        
        totalKE *= 0.5
        
        #remember potential and kinetic energies to be plotted
        self.histPE.append(totalPE)
        self.histKE.append(totalKE)
        
        # We've advanced 1 time step
        self.steps += 1
        self.t += self.dt
    
    def cools(self):
        """
        Cools the system by reducing its kinetic energy.
        """
        N = self.N
        
        for i in range(N):
            self.vx[i] *= self.cool_factor
            self.vy[i] *= self.cool_factor

            
    def saveAnimation(self, filename="LJ"):
        """
        Creates an animation of our Lennard-Jones system and saves it to [filename].gif
        """
        
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        points, = ax.plot(self.x, self.y, 'o', markersize=8)
        title = ax.set_title("default starting title")
        
        def frame(_=0):
            """
            Advances the animation.
            """
            
            # Firstly, and most importantly, display the progress bar.
            x = int(np.floor(32 * self.t / self.tmax) + 1)
            print ("[" + "████████████████████████████████"[:x-1] + "▄" * (x < 32) + "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"[x:] + "]  ", end="\r")
            
            # Advance simulation by dispEvery time steps
            for i in range(self.dispEvery):
                self.advance()
            
            # Enter new particle positions
            points.set_data(self.x, self.y)
            
            title = ax.set_title("t = {:0.2f}".format(self.t))
            
            return points, title
        
        ani = FuncAnimation(fig, frame, frames = int((self.tmax//self.dt)//self.dispEvery),
                            interval = 20, blit = True)
        
        ani.save(filename+".gif")
        plt.close()
        
        return self.histPE, self.histKE