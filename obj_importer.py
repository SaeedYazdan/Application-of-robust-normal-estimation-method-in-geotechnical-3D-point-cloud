import numpy as np

def obj_importer0(name):
    obj=open(name).read()


    loc1=obj.find("model")
    obj=obj[loc1+1 :]
    
    loc2=obj.find("vt")
    obj=obj[ : loc2]
    sec=obj.split('v')
    sec=sec[1:]
    total=np.zeros((len(sec),3))
    
    #for i in range(10):
    for i in range(len(sec)):
        
        x=sec[i].split()
        total[i,:]=[ float(x[0]), float(x[1]), float(x[2])]
        #print(x)
    
    return total
