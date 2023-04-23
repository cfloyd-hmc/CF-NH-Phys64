import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import total_ordering

@total_ordering #allows us to only implement lt and eq, imply gt, etc.
class Disk:
    def __init__(self, x:np.ndarray, v:np.ndarray, mass:float=3, 
                 radius:float=.5, charge:float=1.602e-3):
        self.m = mass
        self.r = radius
        self.x = np.asarray(x) #x example: array([2, 4])
        self.v = np.asarray(v) #v example: array([1.2, -2])
        self.nDim = len(self.x)
        if self.nDim != len(self.v):
            raise Exception("Dimensions of velocity and position lists do not match.")
        self.q = charge
    
#     def forceFrom(self, other, L):
#         #calculates the force vector on self because of other
#         r = self.distFrom(other, L)
#         return (self.COUL_FACTOR * self.q * other.q) * r / (np.linalg.norm(r)**3)
    
    def distFrom(self, other, L):
        r = self.x - other.x # not to be confused with self.r, the radius of a particle
        r[r > L/2] -= L
        r[r < -L/2] += L
        return r
    
    def overlapWith(self, other, L):
        return self.distFrom(other, L) < (self.r + other.r)
    
    def advance(self, dt:float, L, F=0, collidingDisk=False):
        # apply old velocity (update position)
        self.x += self.v * dt
        self.x = self.x % L
        
        # apply force (update velocity)
        self.v += F * (dt / self.m)
        
        #fix collision (?)
        if collidingDisk: #note,this code is definitely unfinished. :)
            self.resolveCollision(self, collidingDisk)
    
    def resolveCollision(self, collidingDisk):
        totalMass = self.m + collidingDisk.m
        
        # Initial velocities of the two disks
        u1 = self.v
        u2 = collidingDisk.v
        
        self.v = ((self.m - collidingDisk.m) / totalMass) * u1 + (2 * collidingDisk.m / totalMass) * u2
        
        collidingDisk.v = (2 * self.m / totalMass) * u1 + ((collidingDisk.m - self.m) / totalMass) * u2
        
        #self.v *= -1 #this is not great, but it's what we got right now.
        
        
    #allowing comparisons between disk
    def __lt__(self, other):
        return self.x[0] < other.x[0]
    
    def __eq__(self, other):
        return self.x[0] == other.x[0]
    
    #explicitly allows statements like "if p: ..."
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
                updateGraphsEvery:int=5, doCollisions=True, potentialType="Coul"):
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
        
        #set potential energy function depending on user input
        if potentialType == "Coul":
            self.COUL_FACTOR = 8.988e9
            self.forceBetween = lambda p1, p2: self._CoulForce(p1, p2)
            self.potentialBetween = lambda p1, p2: self._CoulPotential(p1, p2)
        elif potentialType == "Lenn":
            self.eps = 500 #epsilon, from Lennard-Jones formula
            self.sig = 15 #sigma, from Lennard-Jones formula
            self.forceBetween = lambda p1, p2: self._LennForce(p1, p2)
            self.potentialBetween = lambda p1, p2: self._LennPotential(p1, p2)
        else:
            print("just so you know, forces and potential energies are zero. Hope you wanted that.")
            self.forceBetween = lambda p1, p2: 0
            self.potentialBetween = lambda p1, p2: 0
    
    def _CoulForce(self, p1, p2): #TODO: change code to not need force between particles. 
                #This will also allow user-definined potential functions, which would be great!
        #given two particle IDs/indices, returns the force between them
        if p1 == p2: # in future, maybe change to if p1.friendsWith(p2) and
                     # have particles have a friends list who they don't push
            return 0
        rVec = self.particles[p1].distFrom(self.particles[p2], self.L)
        r = np.linalg.norm(rVec)
        F = self.COUL_FACTOR * self.particles[p1].q * self.particles[p2].q * rVec / r**3
        return F
    def _CoulPotential(self, p1, p2):
        #given two particle IDs/indices, returns the potential between them
        if p1 == p2:
            return 0
        r = np.linalg.norm(self.particles[p1].distFrom(self.particles[p2], self.L))
        V = self.COUL_FACTOR * self.particles[p1].q * self.particles[p2].q / r
        return V
    def _LennForce(self, p1, p2):
        if p1==p2:
            return 0
        rVec = self.particles[p1].distFrom(self.particles[p2], self.L)
        r = np.linalg.norm(rVec)
        F = 24 * self.eps * (2*(self.sig/r)**12-(self.sig/r)**6) * rVec / r**3
        return F
    def _LennPotential(self, p1, p2):
        if p1 == p2:
            return 0
        r = np.linalg.norm(self.particles[p1].distFrom(self.particles[p2], self.L))
        V = -4*self.eps*((self.sig/r)**12-(self.sig/r)**6)
        return V
    
    def nextFrame(self):
        
        #calculate forces in advance
        forces = np.zeros_like(self.particlePositions)
        collisDict = {}
        
        if self.doCollisions:
            #calculate all particles' collisions
            for p1 in range(self.numParticles):
                for p2 in range(self.numParticles):
                    forces[p1] += self.forceBetween(p1, p2)

                    if p1.overlapWith(p2, self.L):
                        collisDict[p1] = p2
        
        #move particles and apply forces afterwards, to allow simultaneity
        forceIter = iter(forces)
        for p in self.particles:
            if self.doCollisions and p in collisDict:
                p.advance(self.dt, self.L, next(forceIter), collistDict[p])
                try:
                    del collisDict[collisDict[p]]
                except:
                    pass
            else:
                p.advance(self.dt, self.L, next(forceIter))
        self.t += self.dt
    
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
            for p2 in range(self.numParticles):
                PE += self.potentialBetween(p1, p2) / 2 #divide by 2 becaue double-counting
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
    
    #idea: makeCopy() function that makes an identical experiment - might be useful to
    #let us go to further times or something? idk
    
    def getKEs(self):
        #Returns a list of the particles' potential energies in the current frame.
        return [p.KE for p in self.particles]

    def showAnimation(self, addlTitle=""):
        
        fig, (ax1, ax2) = plt.subplots(1, 2)              # create the figure
        ax1.set_xlim(0,self.L)              # and adjust axes limits and labels
        ax1.set_ylim(0,self.L)
        ax1.set_title(self.t)
        #TODO: change ax2 axes, title
        HIST_BINS = np.linspace(0, 1000, 100)
        self.updatectr = 0
        
        xvar = np.linspace(0.1,self.L-0.1,self.numParticles) #temporary variable
        points, = ax1.plot(xvar,np.ones_like(xvar), 'o')

        _, _, bar_container = ax2.hist(self.getKEs(), HIST_BINS, lw=1,
                              ec="yellow", fc="green", alpha=0.5)
        
        def frame(_):
            #animate
            points.set_data(np.transpose(self.particlePositions))
            title = ax1.set_title(addlTitle + "t = {:0.2f}".format(self.t))
            
            if self.updatectr % self.updateGraphsEvery == 0:
                
                #histogram
                n, _ = np.histogram(self.getKEs(), HIST_BINS)
                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)
                
                #progress bar
                x = int(np.floor(32*self.t/self.tmax)+1)
                print ("[" + "█████████████████████████████████"[:x] + "▄"*(x<=32) + 
                       "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"[x:] + "]  ", end="\r")
            
            #update
            self.nextFrame()
            
            
            self.updatectr += 1
            return points, bar_container.patches, title 

        #somewhat glitchy display. IDK how best to fix.
        ani = FuncAnimation(fig, frame, np.arange(self.t,self.tmax,self.dt), 
                            interval=self.dt*1000/self.animSpeed, blit=True,
                           repeat=False)
        print("bouta save animation...")
        ani.save("particleAnimation.gif")
        plt.close()
        print("")
        print("finished animating!")
        
        
    def showAnimation1(self, addlTitle=""):
        
        print("initializing experiment... ")
        
        fig, axs = plt.subplot_mosaic(
            """
            AABC
            AADE
            """)              # create the figure
        axs['A'].set_xlim(0,self.L)              # and adjust axes limits and labels
        axs['A'].set_ylim(0,self.L)
        axs['A'].set_title(self.t)
        #TODO: change ax2 axes, title
        HIST_BINS = np.linspace(0, 1000, 20)
        self.updatectr = 0

        xvar = np.linspace(0.1,self.L-0.1,self.numParticles) #temporary variable
        points, = axs['A'].plot(xvar,np.ones_like(xvar), 'o')

        _, _, bar_container = axs['B'].hist(self.getKEs(), HIST_BINS, lw=1,
                              ec="yellow", fc="green", alpha=0.5)

        print("starting simulation... ")
        def frame(_):
            #animate
            points.set_data(np.transpose(self.particlePositions))
            title = axs['A'].set_title(addlTitle + "t = {:0.2f}".format(self.t))

            if self.updatectr % self.updateGraphsEvery == 0:
                #histogram
                n, _ = np.histogram(self.getKEs(), HIST_BINS)
                for count, rect in zip(n, bar_container.patches):
                    rect.set_height(count)

                #progress bar
                x = int(np.floor(32*self.t/self.tmax)+1)
                print ("[" + "████████████████████████████████"[:x-1] + "▄"*(x<32) + "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"[x:] + "]  ", end="\r") #idea: create a dynamically sized progress bar depending on total number of steps that'll be taken, including maybe a 2D one that fills up intelligently :)
                
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
        