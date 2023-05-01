import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import total_ordering

#TODO: integrate periodic and nonperiod BCs into one file?
#

@total_ordering #allows us to only implement lt and eq, imply gt, etc.
class Disk:
    def __init__(self, x:np.ndarray, v:np.ndarray, mass:float=3, 
                 radius:float=5.0, charge:float=1.0, a=0):
        self.m = mass
        self.r = radius
        self.q = charge
        self.x = np.asarray(x) #x example: array([2, 4])
        self.v = np.asarray(v) #v example: array([1.2, -2])
        self.a = np.asarray(a) if a else np.zeros_like(x) #by default, zeros
        self.nDim = len(self.x)
        if self.nDim != len(self.v):
            raise Exception("Dimensions of velocity and position lists do not match.")
    
    def rVecFrom(self, other):
        rVec = self.x - other.x
        return rVec
    
    def overlapWith(self, other):
        return np.linalg.norm(self.rVecFrom(other)) < (self.r + other.r)
    
    def advance(self, dt:float, L, F):
        
        oldx = np.copy(self.x)
        
        # update position, assuming no wall hit
        self.x += (self.v) * dt + (self.a/2) * (dt**2)
        
        # fix wall hits
        for i in range(self.nDim):
            if self.x[i] > L and self.v[i] > 0:
#                 print("right wall hit",self.x,self.v)
#                 self.x[i] = (2*L - self.x[i]) % L
                self.v[i] *= -1
            elif self.x[i] < 0 and self.v[i] < 0:
#                 print("left wall hit",self.x,self.v)
#                 self.x[i] *= -1
                self.v[i] *= -1
        
#         wallhits = self.x[self.x % L != self.x]
#         if wallhits: print(wallhits)
#         wallhits = 2*L - wallhits
        
        # update acceleration
        olda = np.copy(self.a)
        self.a = F/self.m
        
        # update velocity
        self.v += (olda+self.a)/2 * dt
    
    #allowing comparisons between disks, bool checking
    def __lt__(self, other):
        return self.x[0] < other.x[0]
    def __eq__(self, other):
        return self.x[0] == other.x[0]
    def __bool__(self):
        return True
    
    @property
    def speed(self):
        return np.linalg.norm(self.v)
    @property
    def KE(self):
        return (1/2)*self.m*(self.speed**2)
    
class Expt:
    def __init__(self, particles, dt:float=0.1, t_0:float=0, 
                 tmax:float=15, L:float=200, animSpeed:float=1,
                updateGraphsEvery:int=5, doCollisions=True, potentialType="Coul",
                manPotential=None, manForce=None, cool = False):
        # pPositions example: [ [1, 3], [2, 2] ]: two particles, at (1,3) and (2,2)
        
        #set member variables
        self.updateGraphsEvery = updateGraphsEvery
        self.doCollisions = doCollisions
        
        # set time variables
        self.t = t_0
        self.tmax = tmax
        self.dt = dt
        self.animSpeed = animSpeed
        
        # make the particle list
        self.particles = np.asarray(particles)
        self.numParticles = self.particles.size
        self.nDim = self.particles[0].nDim 
        
        #make the box bounds
        self.L = L
        
        # Tells us if the system cools over time
        self.cool = cool
        
        # Number of times that we have cooled the system
        self.coolCount = 0
        
        #set potential energy function depending on user input
        if potentialType == "Coul":
            self.COUL_FACTOR = 10000
            self.forceBetween = lambda p1, p2: self._CoulForce(p1, p2)
            self.potentialBetween = lambda p1, p2: self._CoulPotential(p1, p2)
        elif potentialType == "Lenn":
            self.eps = 10 #10000 #epsilon, from Lennard-Jones formula
            self.sig = 20 #15 #sigma, from Lennard-Jones formula
            self.forceBetween = lambda p1, p2: self._LennForce(p1, p2)
            self.potentialBetween = lambda p1, p2: self._LennPotential(p1, p2)
        elif potentialType == "Man": #manually (user-inputted) force/potential
            self.forceBetween = lambda p1, p2: manForce(p1, p2)
            self.potentialBetween = lambda p1, p2: manPotential(p1, p2)
        else:
            print("just so you know, forces and potential energies are zero. Hope you wanted that.")
            self.forceBetween = lambda p1, p2: 0
            self.potentialBetween = lambda p1, p2: 0
    
    #predefined force, potential functions
    def _CoulForce(self, p1, p2):
        #given two particle IDs/indices, returns the force between them
        if p1 == p2: # in future, maybe change to if p1.friendsWith(p2) and
                     # have particles have a friends list who they don't push
            return 0
        rVec = self.particles[p1].rVecFrom(self.particles[p2])
        r = np.linalg.norm(rVec)
        F = self.COUL_FACTOR * self.particles[p1].q * self.particles[p2].q * rVec / r**3
        return F
    def _CoulPotential(self, p1, p2):
        #given two particle IDs/indices, returns the potential between them
        if p1 == p2:
            return 0
        r = np.linalg.norm(self.particles[p1].rVecFrom(self.particles[p2]))
        V = self.COUL_FACTOR * self.particles[p1].q * self.particles[p2].q / r
        return V
    def _LennForce(self, p1, p2):
        if p1==p2:
            return 0
        rVec = self.particles[p1].rVecFrom(self.particles[p2])
        r = np.linalg.norm(rVec)
        #F = 24 * self.eps * (2*(self.sig/r)**12-(self.sig/r)**6) * rVec / r**3
        
        F = 24 * self.eps * (-2 * (self.sig / r)**12 + (self.sig / r)**6) * (-rVec) / r**2
        return F
    def _LennPotential(self, p1, p2):
        if p1 == p2:
            return 0
        r = np.linalg.norm(self.particles[p1].rVecFrom(self.particles[p2]))
        V = 4*self.eps*((self.sig/r)**12-(self.sig/r)**6)
        return V
    
    def nextFrame(self):
        
        #calculate forces in advance
        forces = np.zeros_like(self.particlePositions)
        collisDict = {}
        
        #detect and store
        for p1 in range(self.numParticles):
            for p2 in range(p1):
                f = self.forceBetween(p1, p2)
                forces[p1] += f
                forces[p2] -= f
                if self.doCollisions and self.particles[p1].overlapWith(self.particles[p2]):
                    collisDict[p1] = self.particles[p2]
        forceIter = iter(forces)
        
        for p in range(self.numParticles):
            if self.doCollisions and p in collisDict: 
                self.resolveCollision(p, collisDict[p])
            self.particles[p].advance(self.dt, self.L, next(forceIter))
        
        self.t += self.dt
    
    def resolveCollision(self, i, j):
        """
        Adjusts the positions and velocities of two particles that have just collided.
        """
        if i == j:
            return
        else:
            p1 = self.particles[i]
            p2 = self.particles[j]
        
            # Move disks so that they no longer overlap
            self.adjustPositions(p1, p2)
        
            totalMass = p1.m + p2.m
        
            # Current velocities of the two disks
            u1 = p1.v
            u2 = p2.v
        
        # Update the velocities
        p1.v = ((p1.m - p2.m) / totalMass) * u1 + (2 * p2.m / totalMass) * u2

        p2.v = (2 * p1.m / totalMass) * u1 + ((p2.m - p1.m) / totalMass) * u2
    
    @property
    def totalKE(self):
        return sum(self.getKEs())
    @property
    def avgKE(self):
        return self.totalKE / self.numParticles
    @property
    def totalPE(self):
        PE = 0
        for p1 in range(self.numParticles):
            for p2 in range(p1):
                PE += self.potentialBetween(p1, p2)
        return PE
    @property
    def avgPE(self):
        return self.totalPE / self.numParticles
    @property
    def totalE(self):
        return self.totalKE + self.totalPE
    @property
    def avgE(self):
        return self.totalE / self.numParticles
    
    @property
    def particlePositions(self):
        return np.array([p.x for p in self.particles])
    
    def getKEs(self):
        #Returns a list of the particles' potential energies in the current frame.
        return [p.KE for p in self.particles]
        
    def showAnimation(self, addlTitle=""):
        
        print("initializing experiment... ")
        
        fig, axs = plt.subplot_mosaic(
            """
            AABB
            AACC
            DDDD
            """)              # create the figure
        axs['A'].set_xlim(0,self.L)              # and adjust axes limits and labels
        axs['A'].set_ylim(0,self.L)
        axs['A'].set_title(self.t)
        #TODO: change ax2 axes, title
        HIST_BINS = np.linspace(0, 1000, 20)
        self.updatectr = 0

        xvar = np.linspace(0.1,self.L-0.1,self.numParticles) #temporary variable
        points, = axs['A'].plot(xvar,np.ones_like(xvar), 'o', markersize=8) #particle positions

        _, _, bar_container = axs['B'].hist(self.getKEs(), HIST_BINS, lw=1,
                              ec="yellow", fc="green", alpha=0.5)

        print("starting simulation... ")
        def frame(_):
            #animate
            points.set_data(np.transpose(self.particlePositions))
            title = axs['A'].set_title(addlTitle + "t = {:0.2f}".format(self.t))

            #graphs
            if self.updatectr % self.updateGraphsEvery == 0:
                #histogram
                n, _ = np.histogram(self.getKEs(), HIST_BINS)
                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)

                #progress bar
                x = int(np.floor(32*self.t/self.tmax)+1)
                print ("[" + "████████████████████████████████"[:x-1] + "▄"*(x<32) + "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"[x:] + "]  ", end="\r") #idea: create a dynamically sized progress bar depending on total number of steps that'll be taken, including maybe a 2D one that fills up intelligently :)
                
                # Cool the system by removing kinetic energy
                if self.cool and self.coolCount < 3:
                    for p in self.particles:
                        p.v *= (1/1.71)
                    self.coolCount += 1
                
            #update
            self.nextFrame()


            self.updatectr += 1
            return points, bar_container.patches, title 

        
        #somewhat glitchy display. IDK how best to fix.
        ani = FuncAnimation(fig, frame, np.arange(self.t,self.tmax,self.dt), 
                            interval=self.dt*1000/self.animSpeed, blit=True,
                           repeat=False)
        ani.save("particleAnimation.gif")
        print("\nfinished animating!")
        plt.close()
