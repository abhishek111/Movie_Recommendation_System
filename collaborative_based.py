from __future__ import division
import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
import math

#import plotly.plotly as py
#import plotly.graph_objs as go


def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

##############################################################
import os
print os.getcwd
ratings=pd.read_csv("ratings.csv");

#u1 = ratings.loc[ratings['userId']== 1]
print "Welcome to the RECOMMEDER SYSTEM"

while True:
    User_Id = int(input("Enter User ID to Login into Profile"))
    u1=ratings.loc[ratings['userId'] == User_Id]
    if( len(u1) != 0 ):
        break
    print "you are not valid User!! Try Again"
u1.index = range(len(u1))
import random
rmovie =random.sample(u1["movieId"],2)
userlist1 = []
for i in  range(len(ratings)):
	if ratings["movieId"][i]==rmovie[0]:
		userlist1.append(ratings["userId"][i])
userlist2 = []
for i in  range(len(ratings)):
	if ratings["movieId"][i]==rmovie[1]:
		userlist2.append(ratings["userId"][i])

inter=list(set(userlist1).intersection(set(userlist2)))
inter = sorted(inter)
columns = ['userId','movie1','movie2']
matrix = pd.DataFrame(columns = columns)
matrix["userId"] = inter
for i in range(len(matrix)):
	for j in range(len(ratings)):
		if ratings["movieId"][j]== rmovie[0] and ratings["userId"][j]==matrix["userId"][i]:
			matrix["movie1"][i] = ratings["rating"][j]
		if ratings["movieId"][j]== rmovie[1] and ratings["userId"][j]==matrix["userId"][i]:
			matrix["movie2"][i] = ratings["rating"][j]

matrix = matrix[matrix.userId != User_Id]

matrix.index = range(len(matrix))
# count = 0
# for j in range(len(ratings)):
#     if ratings["movieId"][j]== rmovie[0] and ratings["userId"][j]==matrix["userId"][count]:
# 		matrix["movie1"][count] = ratings["rating"][j]
# 		count = count+1
# count = 0

# for j in range(len(ratings)):
# 		if ratings["movieId"][j]== rmovie[1] and ratings["userId"][j]==matrix["userId"][count]:
# 			matrix["movie2"][count] = ratings["rating"][j]
# 			count = count+1

#plt.scatter(matrix['movie1'],matrix['movie2'])
#plt.xlabel("movie1")
#plt.ylabel("movie2")
#plt.title("Clustering Of Users")
#plt.show()
X = np.array([matrix['movie1'],matrix['movie2']]) 
X=matrix[['movie1','movie2']].values
cluster = find_centers(X, 3)
print(cluster)

x = u1.loc[u1.movieId==rmovie[0]]['rating']
y = u1.loc[u1.movieId==rmovie[1]]['rating']
min = 1000
clusterNum = -1



for i in range(len(cluster[0])):
    dist = math.sqrt((math.pow((x-cluster[0][i][0]),2))+math.pow((y-cluster[0][i][1]),2))
    if(dist<min):
        min=dist
        clusterNum = i

testuseravg=u1['rating'].mean()

users=[]
for i in range(len(cluster[1][clusterNum])):
    for j in range(len(matrix)):
        if(matrix['movie1'][j]==cluster[1][clusterNum][i][0] and matrix['movie2'][j]==cluster[1][clusterNum][i][1]):
            users.append(matrix['userId'][j])

users=set(users)      #clustered users
users = sorted(users)
meanUsers = []

for i in range(len(users)):
    x = ratings.loc[ratings.userId==users[i]]['rating'].mean()
    meanUsers.append(x)

ratingMul = []
corrBtwUsers = []
sumoftestuser = 0
sumofotheruser = 0
sumnum = 0
for i in range(len(users)):
    commonMovie=sorted(set(ratings.loc[ratings.userId == users[i]]['movieId']).intersection(set(u1['movieId'])))
    userMovie = ratings.loc[ratings.userId == users[i]]
    sumnum = 0
    sumoftestuser = 0
    sumofotheruser = 0
    for j in range(len(commonMovie)):
        sumnum = sumnum + (((userMovie.loc[userMovie.movieId == commonMovie[j]]['rating']-meanUsers[i]).values)[0]*((u1.loc[u1.movieId==commonMovie[j]]['rating']-testuseravg).values))[0]
        sumofotheruser = sumofotheruser + math.pow(((userMovie.loc[userMovie.movieId == commonMovie[j]]['rating']-meanUsers[i]).values)[0],2)
        sumoftestuser = sumoftestuser + math.pow(((u1.loc[u1.movieId==commonMovie[j]]['rating']-testuseravg).values)[0],2)
    z=sumnum/math.sqrt(sumoftestuser*sumofotheruser)
#    print('z',z)
    corrBtwUsers.append(z)

#columns = ['userId','movieId','rating']
#ratingByUsers = pd.DataFrame(columns = columns)
columns = ['userId','corr']
corrdataframe = pd.DataFrame(columns = columns)
corrdataframe['userId'] = users
corrdataframe['corr'] = corrBtwUsers
prediction = []
ratedMovie = []
for i in u1['movieId']:
    sumnum = 0
    sumCorr = 0
    for j in users:
        userMovie = ratings.loc[ratings.userId == j]
        if(any(userMovie.movieId == i)==True):
            sumnum = sumnum + (userMovie.loc[userMovie.movieId==i]['rating']-userMovie['rating'].mean()).values[0]*corrdataframe.loc[corrdataframe.userId==j]['corr'].values[0]
            sumCorr = sumCorr + corrdataframe.loc[corrdataframe.userId==j]['corr'].values[0]
#            print('sumCorr=',sumCorr)
#    print('sumCorr_final=',sumCorr)
    
    if(sumCorr != 0):
        z1 = u1['rating'].mean()
        z2 = sumnum/sumCorr
        prediction.append(z1+z2) 
        ratedMovie.append(i)  
#print(prediction)
actualRating1 = []
for i in ratedMovie:
    x = u1.loc[u1.movieId == i]['rating']
    actualRating1.append(x)

actualRating = []
for i in range(len(actualRating1)):
	x = actualRating1[i].values[0]
	actualRating.append(x)

#plt.scatter(ratedMovie,prediction)
#plt.plot(ratedMovie,prediction,label = 'prediction')
##plt.plot(u1['movieId'],prediction,label = 'prediction')
#plt.scatter(ratedMovie,actualRating)
#plt.plot(ratedMovie,actualRating,label='actual')
#plt.legend(loc='lower right')
#plt.title("Movie Prediction Graph(Collaborative)")
#plt.xlabel("movieId")
#plt.ylabel("rating")
#plt.show()

sumofsqr = 0
rms = 0
for i in range(len(prediction)):
    sumofsqr = sumofsqr + pow((prediction[i] - u1['rating'][i]),2)
rms = math.sqrt(sumofsqr/(len(prediction)))

print(rms)
print "Recommendations for Current User"


notwatchedMovie = []       
for j in users:
        userMovie = ratings.loc[ratings.userId == j]
        userMovie = userMovie[~userMovie['movieId'].isin(u1['movieId'])]
        for i in userMovie['movieId']:
            notwatchedMovie.append(i)
print "reached1"       
notwatchedMovie = set(notwatchedMovie) 
print "reached3"               
for i in notwatchedMovie:
    sumnum = 0
    sumCorr = 0
    for j in users:
        userMovie = ratings.loc[ratings.userId == j]
        #userMovie = userMovie[~userMovie['movieId'].isin(u1['movieId'])]
        if(any(userMovie.movieId == i)==True):
            sumnum = sumnum + (userMovie.loc[userMovie.movieId==i]['rating']-userMovie['rating'].mean()).values[0]*corrdataframe.loc[corrdataframe.userId==j]['corr'].values[0]
            sumCorr = sumCorr + corrdataframe.loc[corrdataframe.userId==j]['corr'].values[0]
#            print('sumCorr=',sumCorr)
#    print('sumCorr_final=',sumCorr)
    
    if(sumCorr != 0):
        z1 = u1['rating'].mean()
        z2 = sumnum/sumCorr
        prediction.append(z1+z2) 
        ratedMovie.append(i)

print "reached2"
columns = ['movieId','prediction']
recommendation_List = pd.DataFrame(columns = columns)
recommendation_List['movieId'] = ratedMovie
recommendation_List['prediction'] = prediction
recommendation_List.index = range(len(recommendation_List))
recommendation_List = recommendation_List.sort_index(by='prediction', ascending = False);

print(recommendation_List.iloc[0:5,1])
movies=pd.read_csv("movies1.csv");
li=[]
for i in recommendation_List['movieId']:
    x = movies.loc[movies.movieId==i]['title']
    li.append(x)
li1 = []
for i in range(len(li)):
    li1.append(li[i].values[0])

recommendation_List['Tilte'] = li1
rl = recommendation_List
maxim = rl['prediction'].max()
minim = recommendation_List['prediction'].min()

for i in range(len(rl)):
    rl['prediction'][i] = ((rl['prediction'][i]-minim)/(maxim-minim))*4+1
    
