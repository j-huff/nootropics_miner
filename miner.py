import praw
import re

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure


import pickle
import numpy as np
from sklearn.cluster import KMeans
from adjustText import adjust_text

# You have to get your own oauth here. Just look up the praw getting started guide

reddit = praw.Reddit(client_id='<client_id>',
                     client_secret='<client secret>',
                     user_agent='miner user agent')


drugs = ['l-theanine','5-htp','nsi-189','alpha-gpc','caffeine','piracetam','modafinil','phenibut','magnesium','gaba','nicotine','semax','adderall','aniracetam','melatonin','phenylpiracetam','ashwagandha','creatine','tyrosine','adrafinil','uridine','curcumin','oxiracetam','selank','kava','pramiracetam','methylphenidate','sulbutiamine','selegiline','ginseng']

f = open('emotions','r')
lines = f.read()
emotions = lines.split('\n')

f = open('moods','r')
lines = f.read()
moods = lines.split('\n')

f = open('disorders','r')
lines = f.read()
disorders = lines.split('\n')

vocab = drugs + emotions + moods

def get_comments(n=1):
	comment_bodies = []
	all_words = []
	comments = []
	i = 0
	for submission in reddit.subreddit('nootropics+stackadvice').top(time_filter='year',limit=n):
		i+=1
		print(repr(i)+"/"+repr(n))
		# print(submission.title)
		all_comments = submission.comments.list()
		# print(all_comments)
		# print(all_comments[0].body)
		comments += all_comments;

	return comments;

def sentences_from_comments(comments):
	sentences = []
	for comment in comments:
		if not hasattr(comment,'body'):
			continue;
		body = comment.body.lower();
		body = re.sub('\\n\\n','\n',body)
		body = re.sub('\\n',' ',body)
		body = re.sub('\'','',body)
		body = re.sub(r'[\\/]+',' ',body)
		for sentence in body.split('. '):
			sentence = re.sub(r'\.','',sentence)
			sentence = re.sub(r'[\(\)]+','',sentence)
			if len(sentence) > 0:
				sentences.append(filter(len,sentence.split(' ')))
	return sentences;


# Set this to True after your first run
initialized = False

if not initialized:
	comments = get_comments(10000)
	pickle.dump(comments, open( "comments.p", "wb" ) )
	sentences = sentences_from_comments(comments)
	pickle.dump(sentences, open( "sentences.p", "wb" ) )

comments = pickle.load( open( "sentences.p", "rb" ) )
print("Done loading")

model = Word2Vec(comments,min_count=5,iter=5,sg=1)

# words = list(model.wv.vocab)

# sim = []
# query = emotions+moods
# for q in query:
# 	if q in words:
# 		sim.append(model.similarity('happy', q))
# 	else:
# 		sim.append(0)

# idx = np.argsort(sim)
# print(idx)
# sim = [sim[i] for i in idx]
# query = [query[i] for i in idx]


# for i in range(len(query)):
# 	print(query[i]+" : "+repr(sim[i]))


words = drugs
X = model[drugs]

# tmp = []
# moods = disorders + moods + emotions
# for mood in moods:
# 	if mood in model.wv.vocab:
# 		tmp.append(mood)
# moods = tmp

# X = np.zeros((len(drugs),len(moods)))
# for i, drug in enumerate(drugs):
# 	for j, mood in enumerate(moods):
# 		X[i,j] = model.similarity(drug,mood)

pca = PCA(n_components=2)
result = pca.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

texts = []
for i, word in enumerate(words):
	if word in vocab:
		x = result[i, 0]
		y = result[i, 1]

		# For plotting lines between similar words
		# for j, word2 in enumerate(words):
		# 	if word == word2 or word2 not in vocab:
		# 		continue;
		# 	sim = model.similarity(word,word2)
		# 	if sim > .7:
		# 		# print("plotting line")
		# 		x2 = result[j, 0]
		# 		y2 = result[j, 1]
		# 		pyplot.plot([x,x2],[y,y2],color='black',alpha=(sim-.2)*.5)


		if word in drugs:
			colors = ['r','g','b','m','y']
			c = colors[kmeans.labels_[i]]
			pyplot.scatter(x, y,color=c)
		else:
			pyplot.scatter(x, y,color='b')
		
		

		texts.append(pyplot.text(result[i, 0], result[i, 1], word, size=10))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5),lim=1000)
pyplot.title("K-means clustering of Nootropics using word embeddings")
pyplot.show()





# Plotting the word embeddings in 3d

# fig = figure()
# ax = Axes3D(fig)

# pca = PCA(n_components=3)
# result = pca.fit_transform(X)
# m = result;

# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	if word in drugs:
# 		ax.scatter(m[i,0],m[i,1],m[i,2],color='r') 
# 	if word in emotions:
# 		ax.scatter(m[i,0],m[i,1],m[i,2],color='g')
# 	if word in moods:
# 		ax.scatter(m[i,0],m[i,1],m[i,2],color='b') 
# 		# ax.text(m[i,0],m[i,1],m[i,2],  '%s' % (word), size=20, zorder=1, color='k')
# 		# pyplot.scatter(result[i, 0], result[i, 1])

# 		# pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# pyplot.show()


# print(comment_bodies)
