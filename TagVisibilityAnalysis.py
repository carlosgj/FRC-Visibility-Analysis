import numpy as np
import math 
import vector
from matplotlib import pyplot as plt
from matplotlib import patches
import itertools
import multiprocessing
from functools import partial

stepsize = 3.

class Robot(object):
    def __init__(self):
        self.cameras = []
    
class Camera(object):
    def __init__(self):
        self.FoV = 90
        self.location = (0,0)
        self.rotation = 0

def rotate(pt, angle):
    x = pt[0]
    y = pt[1]
    s = math.sin(math.radians(angle))
    c = math.cos(math.radians(angle))
    
    xrot = x*c - y*s
    yrot = x*s + y*c
    
    return (xrot, yrot)

def tagsVisible(robot, robotPosition, robotRotation, tags):
    #Get camera locations and rotations
    cameraPoses = []
    visibleTags = 0
    for c in robot.cameras:
        rot = c.rotation + robotRotation
        loc = rotate(c.location, robotRotation)
        boresight = vector.obj(rho=1, phi=math.radians(rot))
        camPose = (loc[0] + robotPosition[0], loc[1] + robotPosition[1], boresight)
        #print(camPose)
        cameraPoses.append(camPose)
        
        for t in tags:
            #print(t)
            CtoT = vector.obj(x=t[0] - camPose[0], y=t[1]-camPose[1])
            offBoresight = boresight.deltaphi(CtoT)
            offBoresight = math.degrees(offBoresight)
            #print(offBoresight)
            if abs(offBoresight) < c.FoV/2.:
                visibleTags += 1
                
    return visibleTags

def getMinimumVisible(idx, robot=None, tags=None):
    x = idx[0] * stepsize
    y = idx[1] * stepsize
    minVisible = 100
    for rot in range(0, 360, 5):
        val = tagsVisible(robot, (x, y), rot, tags)
        if val < minVisible:
            minVisible = val
    return (idx[0], idx[1], minVisible)

if __name__ == "__main__":
    field = np.zeros((int(27*12/stepsize), int(54*12/stepsize)))
            
    robot = Robot()

    #Add two cameras
    c1 = Camera()
    c1.FoV = 90
    c1.location = (15, 0)
    c1.rotation = 0
    robot.cameras.append(c1)

    c2 = Camera()
    c2.FoV = 90
    c2.location = (15, -15)
    c2.rotation = -45
    #robot.cameras.append(c2)
    
    c3 = Camera()
    c3.FoV = 90
    c3.location = (15, 0)
    c3.rotation = 0
    #robot.cameras.append(c3)

    #Add tags to field
    w = 54.*12.
    h = 27.*12.
    tags = [(0, 0), (0, h), (w/2, 0), (w/2, h), (w, 0), (w, h)]

    #tagsVisible(robot, (0,0), 0, tags)

    fig, axs = plt.subplots(2)
    
    axs[0].scatter(*zip(*tags), color='red')
    axs[0].set_xlim([0, 54*12])
    axs[0].set_ylim([0, 27*12])
    
    robotcenter = (200, 27*6)
    axs[0].plot(np.array([-15, 15, 15, -15, -15])+robotcenter[0], np.array([15, 15, -15, -15, 15])+robotcenter[1])
    
    #Plot cameras
    for cam in robot.cameras:
        x = cam.location[0]+robotcenter[0]
        y = cam.location[1]+robotcenter[1]
        axs[0].scatter(x, y, color='blue')
        
        #boresight
        r = 100
        distantx = x + (r * math.cos(math.radians(cam.rotation)))
        distanty = y + (r * math.sin(math.radians(cam.rotation)))
        
        axs[0].plot([x, distantx], [y, distanty], marker=None, linestyle='dashed', color='green')
        
        #FoV boundaries
        angles = []
        for side in [1, -1]:
            fovBoundary = cam.rotation + side * (cam.FoV/2.)
            angles.append(fovBoundary)
            r = 100
            distantx = x + (r * math.cos(math.radians(fovBoundary)))
            distanty = y + (r * math.sin(math.radians(fovBoundary)))
            axs[0].plot([x, distantx], [y, distanty], marker=None, linestyle='solid', color='green')
        
        arc = patches.Arc((x, y), 2*r, 2*r, theta1=angles[1], theta2=angles[0], color='green')
                     
        axs[0].add_patch(arc)
    
        

    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')

    i = range(0, field.shape[1])
    j = range(0, field.shape[0])

    domain = itertools.product(i, j)

    with multiprocessing.Pool(8) as p:
        results = p.map(partial(getMinimumVisible, robot=robot, tags=tags), domain)
            
    print("Merging results...")
            
    for r in results:
        i = r[0]
        j = r[1]
        val = r[2]
        field[j][i] = val
            
    foo = axs[1].imshow(field, vmin=0, vmax=4)
    fig.colorbar(foo, ax=axs[1])
    plt.show()