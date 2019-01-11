import re, io
import numpy as np

def collada_importer0(name):
    cae=open(name).read()

    i=0
    for match in re.finditer("geometry id=", cae):
        i+=1

    xyz_data={}
    for j in range(0,i):

        loc1=cae.find("float_array id=")
        cae=cae[loc1+1 :]
        loc2=cae.find(">")
        cae=cae[(loc2+1) :]

        locE=cae.find("</float_array>")
        defrag=cae[: locE]
        
        xyz_data["geom_{0}".format(j+1)]=defrag

        if not j==i:
            locN=cae.find("geometry id=")
            cae=cae[locN :]   ############ errorenous
    
    with io.open( "soghra1.txt", "w", encoding = "utf-8") as f:
        f.write(xyz_data["geom_1"])

    f.close()

    with io.open( "soghra2.txt", "w", encoding = "utf-8") as f:
        f.write(xyz_data["geom_2"])

    f.close()

    listWords = xyz_data["geom_1"].split(" ")
    listWords2 = xyz_data["geom_2"].split(" ")

    xyzLen=int(len(listWords)/3)
    matrix = np.zeros((xyzLen,3))
    xyzLen2=int(len(listWords2)/3)
    matrix2 = np.zeros((xyzLen2,3))

    for i in range(xyzLen):
        matrix[i,0] = listWords[3*i+0]   # Populating x data
        matrix[i,1] = listWords[3*i+1]   # Populating y data
        matrix[i,2] = listWords[3*i+2]   # Populating z data
        
    for i in range(xyzLen2):
        matrix2[i,0] = listWords2[3*i+0]   # Populating x data
        matrix2[i,1] = listWords2[3*i+1]   # Populating y data
        matrix2[i,2] = listWords2[3*i+2]   # Populating z data

    total=np.append(matrix,matrix2, axis=0)


    return total

