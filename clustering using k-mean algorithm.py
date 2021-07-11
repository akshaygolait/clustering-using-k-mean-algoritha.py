#importing the libaries
import numpy as np
import matplotlib.pyplot as plt
import pa

#importing the iris dataset with pandas
dataset=pd.read_csv('iris.csv')
x=dataset.iloc[:,[0,1,2,3]].values

#finding the optimum number of cluster for k-means classification
from sklearn.cluster import kmeans
wcss=[]

for i in range(1,11):
    kmeans=kMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#plotting the results onto a line graph,allowing us to observe 'The elbow'
    plt.plot(range(1,11),wcss)
    plt.title('The elbow method')
    plt;xlabel('number of clusters')
    plt.ylabel('wcss') # within cluster sum of squares
    plt.show()


#Applying Kmeans to the dataset/ Creating the Kmeans classifier
    kmeans=kmeans(n_clusters=3, init='k-means++',max_iter=300, n_init=10,random_state=0)
    y_kmeans=kmeans.fit_predict(x)

#visualising the clusters
    plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100, c='red',label='iris-setosa')

    plt.scatter(x[y_means==1,0],x[y_kmeans==1,1], s=100, c='blue',label='iris-versicolour')

    plt.scatter(x[y_means==2,0].x[y_kmeans==2,1],s=100 ,c=' green',label='iris-virginica')

    #ploting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.clusters_[:,1],s=100, c='yellow',label='centroids')

    plt.legend()

    plt.show()
