import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model, datasets

numofdata=300
numofout=50
x=np.zeros(numofdata+numofout)
y=np.zeros(numofdata+numofout)
z=np.zeros(numofdata+numofout)
# as column vectors
for i in range(numofdata):
    x[i]=np.random.uniform(10,-10)
    z[i]=np.random.uniform(10,-10)
    y[i]=np.random.uniform(-5, 5)

for i in range(numofout):
    x[i+numofdata]=np.random.uniform(-10,10)
    y[i+numofdata]=10*np.random.uniform(-2, 2)
    z[i+numofdata]=10*np.random.uniform(-2, 2)



def MahalanDist3D(x, y, z):
    leng=len(x)
    
    data=np.zeros((leng,3))
    data[:,0]=x
    data[:,1]=y
    data[:,2]=z
    covariance3D = np.cov(data, rowvar=0)

    inv_covariance_xyz = np.linalg.inv(covariance3D)
    
    xyz_mean = np.mean(x),np.mean(y),np.mean(z)
    x_diff = np.array([x_i - xyz_mean[0] for x_i in x])
    y_diff = np.array([y_i - xyz_mean[1] for y_i in y])
    z_diff = np.array([z_i - xyz_mean[2] for z_i in y])
    diff_xyz = np.transpose([x_diff, y_diff, z_diff])
    
    md = []
    for i in range(len(diff_xyz)):
        l1=np.dot(np.transpose(diff_xyz[i]),inv_covariance_xyz)
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xyz[i]),inv_covariance_xyz),diff_xyz[i])))
    return md



def MD_removeOutliers3D(x, y, z):
    MD = MahalanDist3D(x, y, z)
    threshold = np.mean(MD) * 1.5 # adjust 1.5 accordingly 
    nx, ny, nz, outliers = [], [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(x[i])
            ny.append(y[i])
            nz.append(z[i])
        else:
            outliers.append(i) # position of removed pair
    return (np.array(nx), np.array(ny), np.array(nz), np.array(outliers))


x1,y1,z1,ol=MD_removeOutliers3D(x,y,z)


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(x1, y1, z1, c='r', cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.scatter(x[ol], y[ol], z[ol], c='b', cmap=plt.cm.Set1, edgecolor='k', s=40)

plt.show()
