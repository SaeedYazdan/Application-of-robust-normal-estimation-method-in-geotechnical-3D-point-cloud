from numba import jit
import numpy as np
from scipy import spatial as sp
from timeit import default_timer as timer
from sklearn.neighbors import KDTree



####################################################################################################


def classifier0(Colors):
    Colors[Colors<0.05]=0
    Colors[Colors>=0.05]=1
    ex=np.count_nonzero(Colors == 1)
    print(" number of points which were filtered out"+str(ex))
    return Colors
  
      
@jit
####################################################################################################
def region_growing2(normals, curvature, total, oriCurv, k):
    print(" Performing region growing ...")
    tree=KDTree(total)
    distances, indices =tree.query(total, 30)
    origCurv= np.zeros((len(oriCurv),2))
    for i in range(len(oriCurv)):
        origCurv[i,:]=[i, oriCurv[i]]
    sorCurv=origCurv[origCurv[:,1].argsort()]
    grand_nei_memory=[]
    grand_nei=[]
    plane_normal0=[]
    plane_seed=[]
    number_of_plane=0
    #for point0 in range(10):
    for point0 in range(len(indices)):
        point=sorCurv[point0,0]
        point=int(point)        
        if (point not in grand_nei_memory): 
            nucli_nei=[]
            for nei in range(k): 
                loc=indices[point, nei]
                nucli_nei.append(loc)
            ###########################################
            mean=np.mean(total[indices[point, :]],axis=0)###################
            data_adjust=total[indices[point, :]]-mean#######################
            matrix = np.cov(data_adjust.T)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)        
            sort = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sort]
            eigenvectors = eigenvectors[:,sort]
            aver=eigenvectors[:,2]
            ###################################################
            total_nei=nucli_nei
            nei_of_nei=nucli_nei[1:]
            new_nei_of_nei=nei_of_nei
            if len(nucli_nei)>3:
                while len(new_nei_of_nei)>0:
                    new_nei_of_nei=[]
                    for nei1 in nei_of_nei:
                        nap=0
                        for nei2 in range(1,len(indices[0])):
                            loc=indices[nei1,nei2]
                            if nap < 5:
                                val=abs(np.dot(normals[loc],aver))
                                valN=val/(np.linalg.norm(normals[point])*np.linalg.norm(aver))
                                if ((valN >min_dot) and (loc not in grand_nei_memory) and  (loc not in new_nei_of_nei ) and ( loc not in total_nei )):
                                    Loc_av=np.sum(total[total_nei], axis=0)/len(total_nei)
                                    axel=total[loc,:]-Loc_av
                                    CDi=np.dot((axel).T,aver)
                                    CDts=[]
                                    for dot in total_nei:
                                        CDt=np.dot((total[dot,:]-Loc_av).T,aver)
                                        CDts.append(CDt)
                                    #medi=np.median(CDts)
                                    CDtsnp=np.asarray(CDts)
                                    #######################################################################################################################
                                    mea=np.mean(CDts,axis=0)
                                    standD=np.std(CDts, dtype=np.float64)
                                    ########################################
                                    CDth=mea+2*(standD)
                                    #CDth=medi+2*(qnCD)
                                    if CDi < CDth:
                                        new_nei_of_nei.append(loc)
                                        nap = nap+ 1
                                        #print(len(new_nei_of_nei) , end=' ')
                    total_nei.extend(new_nei_of_nei)
                    new_nei_of_nei=list(set(new_nei_of_nei))
                    nei_of_nei=new_nei_of_nei
                    plane_normal=normals[total_nei,:]
                    ##########################################################################
                    mean=np.mean(total[total_nei,:],axis=0)#######################
                    data_adjust=total[total_nei,:]-mean###########################
                    matrix = np.cov(data_adjust.T)
                    eigenvalues, eigenvectors = np.linalg.eig(matrix)
                    sort = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[sort]
                    eigenvectors = eigenvectors[:,sort]
                    normal=eigenvectors[:,2]
                    normals[point,:]=normal
                    aver=normal
                    ##########################################################################
                    grand_nei_memory.extend(total_nei)
                    grand_nei_memory=list(set(grand_nei_memory))
                    print("hey+++++++++++++++: "+str(len(new_nei_of_nei))+"   z---------------: "+str(len(total_nei)))
                    
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


def main():

    file_to_read="model-Copy.obj"
    k=10
    min_dot=0.97
    min_plane=20
    
    start = timer()

    if file_to_read.endswith('.dae'):
        print("Openning dae file")
        import collada_importer as ci
        total=ci.collada_importer0(file_to_read)
    elif file_to_read.endswith('.obj'):
        print("Opening obj file")
        import obj_importer as ob
        total=ob.obj_importer0(file_to_read)

    else:
        print("Not supported format")
        exit()


    print("The data will be uploaded from previous sessions")
    normals=np.loadtxt("normal.out", delimiter=',')
    curvatures=np.loadtxt("curvatures.out", delimiter=',')
    mod_curv=classifier0(curvatures)


    plane_normal, plane_seed, grand_nei=region_growing2(normals, mod_curv, total, curvatures, k)
    np.savetxt('Pnormal3.out', plane_normal, delimiter=',')
    #np.savetxt('Gnei.out', grand_nei, delimiter=',')

    try:
            os.unlink('Gnei3.txt')
    except:
            print("could not delete")
    try:
            os.unlink('Pseed3.txt')
    except:
            print()
    try:
            np.savetxt('Gnei3.out', grand_nei, delimiter=',')
    except:
            print()
            
    with open('Gnei3.txt', 'w') as f:
            for item in grand_nei:
                f.write("%s\n" % item)

    with open('Pseed3.txt', 'w') as f:
            for item in plane_seed:
                f.write("%s\n" % item)
        
    dt = timer() - start

    print ("Mandelbrot created by numba in %f s" % dt)
