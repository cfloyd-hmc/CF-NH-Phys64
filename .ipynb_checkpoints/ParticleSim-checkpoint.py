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
        self.COUL_FACTOR = 8.988e9
    
    def forceFrom(self, other, L):
        #calculates the force vector on self because of other
        r = self.x - other.x
        r[r > L/2] -= L
        r[r < -L/2] += L
        return (self.COUL_FACTOR * self.q * other.q) * r / (np.linalg.norm(r)**3)
    
    def advance(self, dt:float, L, F=0):
        # apply old velocity (update position)
        self.x += self.v * dt
        self.x = self.x % L
        
        # apply force (update velocity)
        self.v += F * (dt / self.m)
    
    #allowing comparisons between disk
    def __lt__(self, other):
        return self.x[0] < other.x[0]
    
    def __eq__(self, other):
        return self.x[0] == other.x[0]
    
    @property
    def speed(self):
        return np.linalg.norm(self.v)
    
    @property
    def KE(self):
        return (1/2)*self.m*(self.speed**2)

class Expt:
    def __init__(self, particles, dt:float=0.1, t_0:float=0, 
                 tmax:float=15, L:float=200, animSpeed:float=1,
                updateGraphsEvery:int=5):
        # pPositions example: [ [1, 3], [2, 2] ]: two particles, at (1,3) and (2,2)
        
        # set time variables
        self.t = t_0
        self.tmax = tmax
        self.dt = dt
        self.animSpeed = animSpeed
        self.updateGraphsEvery = updateGraphsEvery
        
        # make the particle list
        self.particles = np.asarray(particles)
        self.numParticles = self.particles.size
        self.nDim = self.particles[0].nDim 
        
        #make the box bounds
        self.L = L
    
    def forceBetween(self, p1, p2):
        #given two particle IDs/indices, returns the force between them
        if p1 == p2: # in future, maybe change to if p1.friendsWith(p2) and
                     # have particles have a friends list who they don't push
            return 0
        else:
            return self.particles[p1].forceFrom(self.particles[p2], self.L)
    
    def nextFrame(self):
        
        #calculate forces in advance
        forces = np.zeros_like(self.particlePositions)
        
        #calculate all particles' interactions
        for p1 in range(self.numParticles):
            for p2 in range(self.numParticles):
                if p1 != p2:
                    forces[p1] += self.forceBetween(p1, p2)
        
        #move particles and apply forces afterwards, to allow simultaneity
        forceIter = iter(forces)
        for p in self.particles:
            p.advance(self.dt, self.L, next(forceIter))
        self.t += self.dt
    
    @property
    def totalKE(self):
        return sum(p.KE for p in self.particles)
    
    @property
    def avgKE(self):
        return self.totalKE / self.numParticles
    
    @property
    def particlePositions(self):
        return np.array([p.x for p in self.particles])
    
    #idea: makeCopy() function that makes an identical experiment - might be useful to
    #let us go to further times or something? idk
    
    def getKEs(self):
        """
        Returns a list of the particles' potential energies in the current frame.
        """
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
                print ("[" + "████████████████████████████████"[:x] + "▄"*(x<32) + "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"[x:] + "]  ", end="\r")
            
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