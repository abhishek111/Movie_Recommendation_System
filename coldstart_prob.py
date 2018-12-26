from __future__ import division
import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
import math
# import plotly.plotly as py
# import plotly.graph_objs as go
movies=pd.read_csv("movies1.csv");
print "Welcome to the RECOMMEDER SYSTEM"

movies = movies.fillna(0)

#code for extracting user 1
count = 0
rating=pd.read_csv("ratings.csv")
max_movieId = movies["movieId"].max()
a = [[0 for x in range(2)] for y in range(max_movieId+1)]
for i in range(len(rating)):
    a[rating["movieId"][i]][0]= a[rating["movieId"][i]][0]+rating["rating"][i]
    a[rating["movieId"][i]][1]= a[rating["movieId"][i]][1]+1
    #print a[rating["movieId"][i]][0]
    #print count
    #count = count+1

mean=[0 for x in range(max_movieId+1)]
for i in rating["movieId"]:
    if(a[i][0]!=0 and a[i][1]!=0):
        mean[i]=(a[i][0]/a[i][1])
    

sq = [0 for x in range(max_movieId+2)] 
k=0   
for i in rating["movieId"]:
    if (mean[i]!=0):
        sq[i]=sq[i]+pow((rating['rating'][k]-mean[i]),2)
        k=k+1
          
min=2000000
rootindex=0
for i in rating["movieId"]:
    if(sq[i]<min and sq[i]!=0):
        min=sq[i];
        rootindex=i;
#index_new=rootindex
print rootindex
print "what about movie",
print movies.loc[movies.movieId==rootindex]['title'].values[0];
for l in range(6):
    index_new = rootindex
    choice = raw_input("Enter L if u like the movie, H for hate and U if u don't know about the movie")

    if choice == 'L':
        temp=rating.loc[rating.movieId==index_new]
        user_subset = temp.loc[temp.rating>3]
    elif choice == 'H':
        temp=rating.loc[rating.movieId==index_new]
        user_subset = temp.loc[temp.rating<=3]
    else:
        user_subset=rating.loc[rating.movieId != index_new]
        #ranIndex = np.random.choice(user_subset.index,200)
        #user_subset = user_subset.loc(user_subset.index.isin(ranIndex))
    #print user_subset
    #input()
    movy = [[0 for x in range(2)]for y in range(max_movieId+1)]
    for i in user_subset["userId"]:
        
       # print"reached1"
        temp = rating.loc[rating.userId==i]
        for j in temp["movieId"]:
            if j != index_new:
                movy[j][0]=movy[j][0]+temp.loc[temp.movieId==j]["rating"].values[0]
                movy[j][1]=movy[j][1]+1
              #print "reached2"
              
    mean=[0 for x in range(max_movieId+1)]
    for i in range(len(movy)):
        #print "reached3"
        if(movy[i][0]!=0 and movy[i][1]!=0):
            mean[i]=(movy[i][0]/movy[i][1])
    if l == 5:
        break
    sq = [0 for x in range(max_movieId+1)]
    #for i in range(len(movy)):
    #    print "reached4"
    #    sq[i]=sq[i]+pow((temp["rating"][i]-mean[i]),2)
    for i in user_subset["userId"]:
        #print "reached4"
        temp = rating.loc[rating.userId==i]
        for j in temp["movieId"]:
            #print "reached5"
            sq[j] = sq[j]+pow((temp.loc[temp.movieId==j]['rating'].values[0] -mean[j]),2)
    min=200000
    index_new=1
    for i in temp["movieId"]:
        #print "reached6"
        if(sq[i]<min and sq[i]!=0):
            min=sq[i]
            index_new=i
        rootindex=index_new
    print "what about movie",
    print movies.loc[movies.movieId==rootindex]['title'].values[0];

maxi=-1
#maxim = max(mean)
#minim = min(mean)
for j in range(5):
    for i in range(len(mean)):
        if(mean[i]>maxi):
                maxi=mean[i]
                index=i
    print movies.loc[movies.movieId==index]["title"].values[0],
    print " ",
    #print (mean[i]-minim)/(maxim-minim)*4+1
    print mean[index]
    mean[index]=-1
    maxi=-1
        
                
                
    
    
    
    
    



            
        
           


    
        
       
  
