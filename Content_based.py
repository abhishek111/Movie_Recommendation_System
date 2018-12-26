import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly.graph_objs as go
movies=pd.read_csv("movies1.csv");
#print(movies)
#print(movies.iloc[:,:])
#print(movies.columns.values)
#print(movies.dtypes)
#print(movies["title"][0:5])
#name_id = movies[['title','movieId']]
#descending_sodium = food_info.sort(["Sodium_(mg)"],ascending=[False])
#def assign(x):
#movies,iloc[0,2]
#df_ = df_.fillna(0)

#working code below: making geners 1

print "Welcome to the RECOMMEDER SYSTEM"

movies = movies.fillna(0)



#code for extracting user 1

rating=pd.read_csv("ratings.csv")
while True:
	User_Id = int(input("Enter User ID to Login into Profile"))
	u1=rating.loc[rating['userId'] == User_Id]
	if( len(u1) != 0 ):
		break
	print "you are not valid User!! Try Again"




for i in range(len(movies)-1):
	x=movies["genres"][i]
	w=x.split("|")
	c=movies.columns.values
	for z in w:
		if(z in c):
			movies[z][i]=1

# print(movies)


# print(u1)

#u1['userId'].count();
#train_data=2*((u1['userId'].count())/3);
# print(train_data)

#u1_train = u1.iloc[0:train_data,:]
# print(u1_train)

#creating a empty dataset

#df = pd.DataFrame(columns=movies.columns.values[3:22])
#print(df)

#result = pd.concat([u1_train, movies], axis=1,join='inner')
#u1=u1.set_index('0')
u1.index = range(len(u1))

u2 = pd.merge(u1, movies, on='movieId')

#print(u2.columns.values[6:24])
u3 = u2[u2.columns.values[6:25]].multiply(u1["rating"], axis="index")

u4 = pd.concat([u2[u2.columns.values[0:3]],u3],axis=1)
no_Of_TrainData = 2*((u4['userId'].count())/3);
train_data = u4.iloc[0:no_Of_TrainData,:]
train_data.to_csv('train.csv', sep=',')
#test_data = u4.iloc[no_Of_TrainData+1:u4['userId'].count()]
#print(train_data.iloc[0:(len(train_data)-1)])

# for j range(32):
# 	for i in range(8926):
# 		if(ul_train["movieId"][j]==movies["moviesId"][i]):

#result1 = result #new dataframe

#result1 = result1['rating']*result1[result1.columns.values[6:21]]

#result2=result1[result1.columns.values[6:21]].multiply(result1["rating"], axis="index")




#result3 = pd.merge(result['userId','movieId','rating'], result2, axis="index")

#result2['userId']=result.userId
#result2['rating']=result.rating
#result2['movieId']=result.movieId

#result1=pd.concat([result[result.columns.values[0:3]],result2],axis=1)
#print(result1)

weight=u3.iloc[0:(len(train_data)),:].sum()/no_Of_TrainData
#print(weight)
weight=pd.Series.to_frame(weight)
weight=weight.transpose();
print(weight)
test_data = u2.iloc[no_Of_TrainData:u4['userId'].count(),:]

test_data.index = range(test_data['userId'].count())
#print(test_data)
sum=0;
pred = []
for i in range(test_data['userId'].count()):
	x=test_data["genres"][i]
	w=x.split("|")
	sum=0
	for y in w:
		sum = sum + weight[y][0]
	pred.append(sum)

		

test_data['predRating'] = pred
test_data.to_csv('out.csv', sep=',')
#print(test_data['predRating'])
# maximum = max(test_data['predRating'])
# minimum = min(test_data['predRating'])
# diff = maximum-minimum
# test_data['predRating'] = ((test_data['predRating']) - minimum)/diff
# test_data['predRating'] = test_data['predRating']*5

# maximum1 = max(test_data['rating'])
# minimum1 = min(test_data['rating'])
# diff1 = maximum1-minimum1
# test_data['rating'] = ((test_data['rating']) - minimum1)/diff1
# test_data['rating'] = test_data['rating']*5

RootMeanSqrt = np.sqrt(((test_data['rating']-test_data['predRating'])**2).mean()) 
print(RootMeanSqrt)
umov = pd.DataFrame(columns=movies.columns.values[0:2])
#print(umov.columns.values)
count = 0
flag = 1
count1=0
for i in movies['movieId']:
	if(count1>100):
		break;
	#print(i)
	for j in u1['movieId']:
		if(j == i):
			flag = 0
	if(flag == 1):
		umov = umov.append(movies.iloc[count,0:3]);
		#print(movies.iloc[count,0:2])
		count1 = count1+1;
	flag = 1;
	count = count+1;
umov.index = range(len(umov))
#print(umov);
pred_new = []
for i in range(len(umov)):
	x=umov['genres'][i]
	w=x.split("|")
	sum=0
	for y in w:
		sum = sum + weight[y][0]
	pred_new.append(sum)
umov['predRating'] = pred_new

umov = umov.sort_index(by='predRating', ascending = False);
umov.index = range(len(umov))
#print(umov)
print "Recommended Movies:"
print(umov.iloc[0:6,1])
# trace1 = go.Scatter(x=test_data['movieId'],y = test_data['predRating'])
# trace2 = go.Scatter(x= test_data['movieId'],y = test_data['rating'])
# data = [trace1, trace2]
# dict(data = data)
plt.scatter(test_data['movieId'],test_data['predRating'])
plt.plot(test_data['movieId'],test_data['predRating'],label = 'prediction')
plt.scatter(test_data['movieId'],test_data['rating'])
plt.plot(test_data['movieId'],test_data['rating'],label='actual')
plt.legend(loc='lower right')
plt.title("Movie Prediction Graph")
plt.xlabel("moviId")
plt.ylabel("ratings")
plt.show()