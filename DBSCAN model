import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

dataset = read_csv('C:/Users/ZHUHO/AppData/Local/Programs/Python/Python36/DBSCAN.csv')
data = dataset.values

def main(data, eps=0.3, min_samples=10):
    db=DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    coreSampleMask=np.zeros_like(db.labels_, dtype=bool)
    coreSampleMask[db.core_sample_indices_]=True
    clusterLabels=db.labels_
    uniqueClusterLabels=set(clusterLabels)
    nclusters=len(uniqueClusterLabels)-(-1 in clusterLabels)

    colors=['red','green','blue','black','gray','#ff00ff','#ffff00']
    markers=['v','^','o','*','x','h','d']
    for i,cluster in enumerate(uniqueClusterLabels):
        print('聚类标签为[]的数据'.format(cluster).center(30,'='))
        clusterIndex=(clusterLabels==cluster)

        coreSamples=data[clusterIndex&coreSampleMask]
        print('核心对象'.ljust(40,'*'))
        print(coreSamples)
        if i==0:
            A1 = coreSamples[:-1]
            A1 = A1.astype(np.float64)
            print(A1.dtype)
            np.save('coresamples(A1).npy', A1)
        elif i==1:
            B1 = coreSamples[:-1]
            B1 = B1.astype(np.float64)
            print(B1.dtype)
            np.save('coresamples(B1).npy', B1)
        elif i==2:
            C1 = coreSamples[:-1]
            C1 = C1.astype(np.float64)
            print(C1.dtype)
            np.save('coresamples(C1).npy', C1)
        else:
            D1 = coreSamples[:-1]
            D1 = D1.astype(np.float64)
            print(D1.dtype)
            np.save('coresamples(D1).npy', D1)
        plt.scatter(coreSamples[:,0], coreSamples[:,1], c=colors[i], marker=markers[i],s=80)



        noiseSamples=data[clusterIndex&(~coreSampleMask)]
        print('非核心对象'.ljust(30,'*'))
        print(noiseSamples)
        if i == 0:
            A2 = noiseSamples[:-1]
            A2 = A2.astype(np.float64)
            print(A2.dtype)
            np.save('noiseSamples(A2).npy', A2)
        elif i == 1:
            B2 = noiseSamples[:-1]
            B2 = B2.astype(np.float64)
            print(B2.dtype)
            np.save('noiseSamples(B2).npy', B2)
        elif i == 2:
            C2 = noiseSamples[:-1]
            C2 = C2.astype(np.float64)
            print(C2.dtype)
            np.save('noiseSamples(C2).npy', C2)
        else:
            D2 = noiseSamples[:-1]
            D2 = D2.astype(np.float64)
            print(D2.dtype)
            np.save('noiseSamples(D2).npy', D2)
        plt.scatter(noiseSamples[:,0], noiseSamples[:,1],c=colors[i], marker=markers[i], s=26)
    plt.show()

main(data,30,30)





