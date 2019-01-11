import re, io, os
import numpy as np
import vtk
from vtk import *
from sklearn.neighbors import KDTree
from functools import reduce
import winsound
import scipy.stats as stats
import pylab as pl
import math
import matplotlib.pyplot as plt
import mplstereonet
import random
import time
import skfuzzy as fuzz
import DetMM as dm

##########################################################################################################################

file_to_read="model-Copy.obj"


k=10
max_curv=0.4
min_dot=0.95
min_plane=20
##########################################################################################################################

class VtkPointCloud:

    def __init__(self, zMin=0, zMax=1, maxNumPoints=1e8):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(1, 255)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(2)
        self.vtkActor.SetMapper(mapper)
        
        # create source
        src = vtk.vtkPointSource()
        #src.SetCenter(0, 0, 0)
        src.SetNumberOfPoints(50)
        src.SetRadius(5)
        src.Update()

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())


    def addPoint(self, point,color):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(color)
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
    

def coplaner(total, indices, k):
    normals=np.zeros((len(indices),3))
    curvatures=np.zeros(len(indices))
    #for point in range(25):
    for point in range(len(indices)):

        #######################################
        neighbors=indices[point,:]
        outMM=dm.DetMM0(total[neighbors,:])
        covar=outMM["cov"]
        ########################################

        eigenvalues, eigenvectors = np.linalg.eig(covar)
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        curvature=abs(eigenvalues[2])/(abs(eigenvalues[0])+abs(eigenvalues[1])+abs(eigenvalues[2]))
        eigenvectors = eigenvectors[:,sort]
        normal=eigenvectors[:,2]
        normals[point,:]=normal
        curvatures[point]=curvature
        if point%100==0:
            print("Accomplishment of: "+str(point)+" points out of: "+str(len(indices)))
    return normals, curvatures

    
def classifier(Colors):
    Colors[Colors<=0.05]=0.02
    Colors[Colors>0.05]=0.5
    Colors[Colors>=0.2]=0.9
    ex=np.count_nonzero(Colors == 0.9)
    print(" number of points which were filtered out"+str(ex))
    return Colors

def spherical_coor(normals):
    angles=np.zeros((len(normals),2))
    print(" Drawing streonet...")
    for i in range(len(normals)):
        
        n=normals[i]
        theta=math.degrees(math.atan (n[1]/n[0]))
        phi=math.degrees(math.acos(n[2]/math.sqrt(n[0]**2+n[1]**2+n[2]**2)))
        angles[i,0]=theta
        angles[i,1]=phi
    return angles


def plotter(data):
    
    fig, ax = mplstereonet.subplots()
    strikes = data[:,0]
    dips = data[:,1]
    cax = ax.density_contourf(strikes, dips, measurement='poles')
    ax.pole(strikes, dips)
    ax.grid(True)
    fig.colorbar(cax)
    plt.show()
        
def normal_mean(normal):
    aver=np.mean(normal, axis=0)
    val=abs(np.dot(normals[point],aver))
    valN=val/(np.linalg.norm(normals[point])*np.linalg.norm(aver))
    return valN

################################################################
def Delaunay(grand_nei, total, cluster_membership):

    print("doing delaunay")
    i=0
    for plane in grand_nei:

        Xp=plane.split(']')
        plane0=Xp[0]
        #print(plane0)
        plane=plane0.split(",")

        #print(plane)
        ############### Creating random color
        color=cluster_membership[i]
        i=i+1
        #color0=[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]

        color0=[0.9*color+0.1, 0.75*color, 0.2*color]
        
        meshActor={}
        boundaryActor={}
        points=vtk.vtkPoints()
        for itom in plane:
            item=int(itom)
            xyz2=[total[item,0], total[item,1], total[item,2]]
            points.InsertNextPoint(xyz2)

        aPolyData = vtk.vtkPolyData()
        aPolyData.SetPoints(points)

        aCellArray = vtk.vtkCellArray()

        boundary = vtk.vtkPolyData()
        boundary.SetPoints(aPolyData.GetPoints())
        boundary.SetPolys(aCellArray)
        delaunay = vtk.vtkDelaunay2D()
        if vtk.VTK_MAJOR_VERSION <= 5:
            delaunay.SetInput(aPolyData.GetOutput())
            delaunay.SetSource(boundary)
        else:
            delaunay.SetInputData(aPolyData)
            delaunay.SetSourceData(boundary)

        delaunay.Update()

        meshMapper = vtk.vtkPolyDataMapper()
        meshMapper.SetInputConnection(delaunay.GetOutputPort())



        meshActor["v{0}".format(i)] = vtk.vtkActor()
        meshActor["v{0}".format(i)].SetMapper(meshMapper)
        meshActor["v{0}".format(i)].GetProperty().SetEdgeColor(0, 0, 1)
        meshActor["v{0}".format(i)].GetProperty().SetColor(color0)
        meshActor["v{0}".format(i)].GetProperty().SetInterpolationToFlat()
        #meshActor["v{0}".format(i)].GetProperty().SetRepresentationToWireframe()

        boundaryMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            boundaryMapper.SetInputConnection(boundary.GetProducerPort())
        else:
            boundaryMapper.SetInputData(boundary)

        boundaryActor["v{0}".format(i)] = vtk.vtkActor()
        boundaryActor["v{0}".format(i)].SetMapper(boundaryMapper)
        boundaryActor["v{0}".format(i)].GetProperty().SetColor(1, 0, 0)

        renderer.AddActor(meshActor["v{0}".format(i)])
        renderer.AddActor(boundaryActor["v{0}".format(i)])

##############################################################################################

def normal_drawer(plane_normal, plane_seed):
    print( " Drawing plane Normals")
    for plane in range(len(plane_normal)):

        pl= plane_seed[plane]
        pl=pl[pl.find('(') :]
        pl=pl[pl.find('[')+1 :]
        pl=pl[: pl.find(']')]
        pt=pl.split(',')
        #print(pt)
        p0x=float(pt[0])
        p0y=float(pt[1])
        p0z=float(pt[2])
        
        pn=plane_normal[plane]
        p0 = [p0x, p0y, p0z]
        p1 = [p0x+pn[0], p0y+pn[1], p0z+pn[2]]


        # Create a vtkPoints object and store the points in it
        points = vtk.vtkPoints()
        #points.InsertNextPoint(origin)
        points.InsertNextPoint(p0)
        points.InsertNextPoint(p1)

        # Create a cell array to store the lines in and add the lines to it
        lines = vtk.vtkCellArray()

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,0)
        line.GetPointIds().SetId(1,1)
        lines.InsertNextCell(line)

        # Create a polydata to store everything in
        linesPolyData = vtk.vtkPolyData()
 
        # Add the points to the dataset
        linesPolyData.SetPoints(points)
 
        # Add the lines to the dataset
        linesPolyData.SetLines(lines)
 
        # Setup actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(linesPolyData)
        else:
            mapper.SetInputData(linesPolyData)
 
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1,0,0)
        renderer.AddActor(actor)

    
####################################################################################################
def region_growing2(normals, indices, curvature, total, oriCurv, k):
    print(" Performing region growing ...")

    origCurv= np.zeros((len(oriCurv),2))
    for i in range(len(oriCurv)):
        origCurv[i,:]=[i, oriCurv[i]]

    sorCurv=origCurv[origCurv[:,1].argsort()]
    grand_nei_memory=[]
    grand_nei=[]
    reg_norm=np.zeros((k,3))
    plane_normal0=[]
    plane_seed=[]
    number_of_plane=0 
    for point0 in range(len(indices)):
        point=sorCurv[point0,0]
        point=int(point)        
        if ((curvature[point]<max_curv) and (point not in grand_nei_memory)): 
            nucli_nei=[]
            for nei in range(k): 
                loc=indices[point, nei]
                if curvature[loc]<max_curv:
                    normalt=normals[loc,:]
                    reg_norm[nei,:]=normalt
                    nucli_nei.append(loc)
            aver=np.mean(reg_norm, axis=0)
            total_nei=nucli_nei
            nei_of_nei=nucli_nei[1:]
            new_nei_of_nei=nei_of_nei
            if len(nucli_nei)>3:
                while len(new_nei_of_nei)>0:
                    new_nei_of_nei=[]
                    for nei1 in nei_of_nei:
                        for nei2 in range(1,k):
                            loc=indices[nei1,nei2]
                            val=abs(np.dot(normals[loc],aver))
                            valN=val/(np.linalg.norm(normals[point])*np.linalg.norm(aver))
                            if (((valN >min_dot) and (loc not in grand_nei_memory)) and (curvature[loc]<max_curv)):
                                new_nei_of_nei.append(loc)
                    total_nei.extend(new_nei_of_nei)
                    total_nei=list(set(total_nei))
                    new_nei_of_nei=list(set(new_nei_of_nei))
                    nei_of_nei=new_nei_of_nei

                    plane_normal=normals[total_nei,:]
                    aver=np.mean(plane_normal, axis=0)
                    grand_nei_memory.extend(total_nei)
                    grand_nei_memory=list(set(grand_nei_memory))
                    
                    #print("attached "+str(len(total_nei))+" points to the plane!")
                    
                if len(total_nei)>min_plane:
                    grand_nei.append(total_nei)
                    number_of_plane+=1
                    print("   number of point:   "+str(point0)+" and the length of the sheath is: "+str(len(total_nei)))   
                    print (" number of plane:   "+str(number_of_plane))
                    plane_seed.append([total[point,:]])
                    plane_normal0.append([aver[0],aver[1],aver[2]])

    print("number of points already included in construction of surfaces: "+str(len(grand_nei_memory)))
    
    return plane_normal0, plane_seed, grand_nei

#########################################################################################################################################


tic=time.time()


if file_to_read.endswith('.dae'):
    print("openning dae file")
    import collada_importer as ci
    total=ci.collada_importer0(file_to_read)
elif file_to_read.endswith('.obj'):
    print("opening obj file")
    import obj_importer as ob
    total=ob.obj_importer0(file_to_read)

else:
    print("not supported format")
    exit()




 
pointCloud = VtkPointCloud()
print(" Finding neighbors...")

tree=KDTree(total)
distances, indices =tree.query(total, k)
total=total[np.where(distances[:,1] != 0)]
tree=KDTree(total)
distances, indices =tree.query(total, k)

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(pointCloud.vtkActor)
renderer.SetBackground(0.5, 0.5, 1)
renderer.ResetCamera()
#renderer.SetActiveCamera(camera);

num_to_show=len(indices)

print(" Finding points normal...")


inp=input("Do you want me to find teh normals again? (y/n) ")

if (inp=='y'):
    print('It might take couple of hours. So be paitient')

    normals, curvatures = coplaner(total, indices[:num_to_show], 5)
    np.savetxt('normal.out', normals, delimiter=',')
    np.savetxt('curvatures.out', curvatures, delimiter=',')
    
elif (inp=='n'):
    print("The data will be uploaded from previous sessions")
    normals=np.loadtxt("normal.out", delimiter=',')
    curvatures=np.loadtxt("curvatures.out", delimiter=',')

else:
    exit()



print(" Calculating coplannarity...")

#mod_curv=classifier(25*curvatures)
mod_curv=curvatures


inp=input("Do you want me to Do the region growing? (y/n) ")

if (inp=='y'):

    plane_normal, plane_seed, grand_nei=region_growing2(normals, indices, mod_curv, total, curvatures, k)

    np.savetxt('Pnormal2.out', plane_normal, delimiter=',')
    #np.savetxt('Gnei.out', grand_nei, delimiter=',')
    try:
        os.unlink('Gnei2.txt')
    except:
        print()
    try:
        os.unlink('Pseed2.txt')
    except:
        print()

    with open('Gnei2.txt', 'w') as f:
        for item in grand_nei:
            f.write("%s\n" % item)

    with open('Pseed2.txt', 'w') as f:
        for item in plane_seed:
            f.write("%s\n" % item)

    
    frequency = 1200  # Set Frequency To 2500 Hertz
    duration = 400  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    exit()
    
elif(inp=='n'):
    plane_normal=np.loadtxt("Pnormal2.out", delimiter=',')
    #grand_nei=np.loadtxt("Gnei.out", delimiter=',')

    #plane_seed=np.loadtxt("Pseed.txt", delimiter=',')
    file = open("Pseed2.txt", "r") 
    plane_seed=file.readlines()
    #

    file = open("Gnei2.txt", "r") 
    gt=file.read()

    grand_nei=[]
    for i in range(len(plane_seed)):
        loc1=gt.find("[")
        gt=gt[loc1 :]
        #print(gt)
        
        loc2=gt.find("]")
        row=gt[loc1+1:loc2]
        gt=gt[(loc2+2) :]
        grand_nei.append(row)
    
else:
    exit()



plane_normal_deg=spherical_coor(plane_normal)
#print(plane_normal_deg)


cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    plane_normal.T, 4, 2, error=0.005, maxiter=1000, init=None)


cluster_membership = np.argmax(u, axis=0)

print(cluster_membership)


Delaunay(grand_nei, total, cluster_membership)

#normal_drawer(0.2*plane_normal, plane_seed)

plotter(plane_normal_deg)

print("ALL done")

"""
plt.hist(curvatures, color = 'blue', edgecolor = 'black',
         bins = int(500))

plt.axis([0,0.2,0,20000])
plt.show()



for i in range(num_to_show):
    pointCloud.addPoint([total[i,0],total[i,1],total[i,2]], 10000*mod_curv[i])


"""



transform = vtk.vtkTransform()
transform.Translate(1.0, 0.0, 0.0)
 
axes = vtk.vtkAxesActor()
#  The axes are positioned with a user transform
axes.SetUserTransform(transform)

#renderer.AddActor(axes)

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.Initialize()

# Begin Interaction

renderWindow.Render()
renderWindowInteractor.Start()

toc=time.time()
print("the total amount of time taken in second: "+str(toc-tic))
