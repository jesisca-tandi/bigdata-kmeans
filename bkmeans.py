### Packages versions:
# Python 3.7.0 (v3.7.0:1bf9cc5093)
# pyspark 2.4.4 hadoop2.7
# scala 2.11.12
# java jdk1.8.0_221

### Input arguments (if not specified, load the default ones)
# 1) location to input documents 
# 2) location of a file containing keywords of interest
# 3) location of a file containing stopwords


import numpy as np
import pandas as pd
import re,os,sys,shutil
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# Load input
if len(sys.argv) == 1:
	docDir = 'data/input/'
	queryFile = 'data/query.txt'
	stopWordsFile = 'data/stopwords.txt'

elif len(sys.argv) != 4:
	print('ERROR: Specify 3 inputs: datafiles location, query words file, and stopwords file')
	exit()

else:
	docDir = sys.argv[1]
	queryFile = sys.argv[2]
	stopWordsFile = sys.argv[3]


####################################################
####################################################
####################################################
###############       Task A        ################
####################################################
####################################################
####################################################



###################### Step 1 ######################
########### Compute TF for each document ###########
####################################################

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Create a function to preprocess texts
# - remove special characters
# - convert to lower cases
def preprocess(x):
	return re.sub('[~`!@#$%^&*()_+-=\\[\\]{}|\\\;:<>,.?/\"\']', '', x.lower())

# Load stopwords, preprocess, and convert to list
stopWords = sc.textFile(stopWordsFile)
stopWords = stopWords.flatMap(lambda x: re.split(r'[^\w]+', preprocess(x))).collect()

# Load documents and then convert to word counts of each document (TF)
# MAP 
# 	input = location to input documents
# 	resulting key, value pairs:
#   	key : <filename, word>
#   	value: <1>
docs = sc.wholeTextFiles(docDir)
docCounts = docs.flatMap(lambda x: [((x[0], word), 1) for word in re.split(r'[^\w]+', preprocess(x[1])) if word not in stopWords])
# REDUCE to word counts within each document, each word
# 	key : <filename, word>
# 	value : <wordcounts>
tf = docCounts.reduceByKey(lambda n1, n2: n1 + n2)




###################### Step 2 ######################
####### Compute TF-IDF for each document ###########
#######  (1 + log (TF)) * ( log (N / DF) )  ########
####################################################

# Get the list of all documents (to compute N)
allDocs = docs.map(lambda x: x[0]).collect()
N = len(allDocs)

# Get the number of documents containing each word
# MAP, output:
# 	key <word>
# 	value <1>  for each document where the word appears
# REDUCE, output: 
# 	key <word> 
#	value <document counts>
df = tf.map(lambda x: (x[0][1], 1)) \
		.reduceByKey(lambda n1, n2: n1 + n2)

# Get the list of all words combined
allWords = df.map(lambda x: x[0]).collect()

# Replicate df-s for all documents
# MAP, output:
# 	key <filename, word> 
#	value <document counts>
df_replicated = df.flatMap(lambda x: [((f, x[0]), x[1]) for f in allDocs])


# Insert 'tf' to indicate this is TF
tf = tf.map(lambda x: (x[0], ('tf', x[1])))
# Insert 'df' to indicate this is DF
df_replicated = df_replicated.map(lambda x: (x[0], ('df', x[1])))


# Create function to compute TF-IDF
def computeTFIDF(x,y):

	if (x[0] != y[0]):

		if x[0] == 'tf':
			tf = x[1]
		if y[0] == 'df':
			df = y[1]
		if x[0] == 'df':
			df = x[1]
		if y[0] == 'tf':
			tf = y[1]

		return (1+np.log10(tf))*(np.log10(N/df))


# MAP-REDUCE, output: 
#	key <filename, word>
#	value <tfidf>
# Steps: combine TF and DF rdds, group based on their keys (document, word), then apply function
tfidf = tf.union(df_replicated) \
	.groupByKey() \
	.mapValues(lambda x: list(x)) \
	.filter(lambda x: len(x[1])>1) \
	.flatMap(lambda x: [(x[0],item) for item in x[1]]) \
	.reduceByKey(lambda x,y : computeTFIDF(x, y))




###################### Step 3 ######################
###### Normalize TF-IDF for each document ##########
####################################################

# Compute normalization factors (ssq) for each document
# MAP-REDUCE, output
#	key <filename> 
#	value <sum of squares>
ssq = tfidf.map(lambda x: (x[0][0], (x[1]**(2)))) \
		.reduceByKey(lambda x,y: x + y)

# Replicate for all words
# MAP, output: 
#	key <filename, word> 
#	value <'ssq', sumofsquares>
ssq_replicated = ssq.flatMap(lambda x: [((x[0], w), ('ssq', x[1])) for w in allWords])


# Add tfidf tag for indication
tfidf = tfidf.map(lambda x: (x[0], ('tfidf', x[1])))

# Create function to compute normalized TF-IDF
def computeNormTFIDF(x,y):

	if (x[0] != y[0]):

		if x[0] == 'tfidf':
			tfidf = x[1]
		if y[0] == 'ssq':
			ssq = y[1]
		if y[0] == 'tfidf':
			tfidf = y[1]
		if x[0] == 'ssq':
			ssq = x[1]

		return tfidf/np.sqrt(ssq)

# Compute normalized TF-IDF
# Steps: 	combine TF-IDF and SSQ (normalization factor)
#			group by key (docid, word) and combine
#			filter only those with both TF-IDF and SSQ
#			re-generate key-value pairs for each word & doc
#			apply function
# MAP-REDUCE, output: key <filename, word> and value <norm_tfidf>
normtfidf = tfidf.union(ssq_replicated) \
	.groupByKey() \
	.mapValues(lambda x: list(x)) \
	.filter(lambda x: len(x[1])>1) \
	.flatMap(lambda x: [(x[0],item) for item in x[1]]) \
	.reduceByKey(lambda x,y : computeNormTFIDF(x, y))

###################### Step 4 ######################
############ Compute the relevance of ##############
#########  each document w.r.t a query. ############
####################################################

# Read query document
# output key: <queryword> and value: <'query'>
query = sc.textFile(queryFile)
query = query.flatMap(lambda l: re.split(r'[^\w]+', preprocess(l)))
n_query = len(query.collect())
query = query.map(lambda x: (x, 'query'))


# Normalize norm-TF-IDF against query's SSQ ||B||
# where ||B|| = sqrt ( #query )
# MAP, output: 
#	key: <word>
# 	value: <filename, 'tfidf', tfidfscore/sqrt(n_query))>
normtfidf_word = normtfidf.map(lambda x: (x[0][1], ('tfidf', x[0][0], x[1]/np.sqrt(n_query))))


# Filter only words being queried and sum over all TF-IDF of these words for each document
# Steps: 	combine normalized TF-IDF and query words
# 			group keys (docid, word) and combine values
#			filter only words being queried
#			expand back TF-IDF scores to key <docid, word> and value <norm TF-IDF>
#			sum all filtered normalized TF-IDF
# MAP-REDUCE, output :
# 	key <filename> 
#	value <relevance>
relevance = normtfidf_word.union(query) \
	.groupByKey() \
	.mapValues(lambda x: tuple(x)) \
	.filter(lambda x: True if 'query' in x[1] else False) \
	.flatMap(lambda x: [(x[0], item) for item in x[1] if 'query' not in item]) \
	.map(lambda x: (x[1][1], x[1][2])) \
	.reduceByKey(lambda x,y: x+y)


###################### Step 5 ######################
########### Sort and get top-N documents ###########
################## use N = 10 ######################

n = 10

relevanceSorted = relevance.takeOrdered(n, lambda x: -x[1])
relevanceSorted = sc.parallelize(relevanceSorted)

# Save output
if os.path.exists('out'):
	shutil.rmtree('out')
relevanceSorted.saveAsTextFile('out')






####################################################
####################################################
####################################################
###############       Task B        ################
####################################################
####################################################
####################################################


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

# Create dataframe from normalized TF-IDF
normtfidf_array = normtfidf \
	.map(lambda x: (x[0][0], Vectors.dense([x[1] if (x[0][1]==w) else 0 for w in allWords]))) \
	.reduceByKey(lambda x,y: x+y)
X = normtfidf_array.toDF(['doc', 'features'])
# print(allWords, file=open('Words.txt', 'w'))


# Run kmeans Clustering - distance measures: (1) euclidean and (2) cosine 
# k from 2 to 9
k_list = range(2, 9)

for dist in ['cosine', 'euclidean']:

	ssq = [] # Sum of Within-cluster error scores
	silh = [] # Silhouette scores
	npoints = [] # Number of datapoints in each cluster
	
	for k in k_list:

		kmeans = KMeans(distanceMeasure=dist, tol=1e-6, maxIter=50).setK(k).setSeed(123)
		kmeansModel = kmeans.fit(X)
		ssq.append(kmeansModel.computeCost(X)) # Save within-cluster error score

		# Compute silhouette 
		clusters = kmeansModel.transform(X)
		evaluator = ClusteringEvaluator()
		silhouette = evaluator.evaluate(clusters)
		silh.append(silhouette)
		
		# Save the number of members / datapoints in each cluster
		npoints.append(clusters.toPandas().prediction.value_counts().values.tolist())


	# Plot error scores and silhouette
	f, ax = plt.subplots(2, figsize=(3,4))
	ax[0].plot(k_list, ssq)
	ax[0].set_xticks(k_list)
	costLabel = {'cosine': 'Sum of distance\n(cosine)', 'euclidean': 'Sum of distance\n(squared euclidean)'}
	ax[0].set_ylabel(costLabel[dist])
	ax[0].set_xlabel('Number of clusters')
	ax[1].plot(k_list, silh)
	ax[1].set_xticks(k_list)
	ax[1].set_ylabel('Silhouette')
	ax[1].set_xlabel('Number of clusters')
	plt.tight_layout()
	plt.savefig('Kmeans_{}.svg'.format(dist))
	plt.close()

	# Plot number of points in each cluster
	dt = pd.DataFrame(npoints)
	dt = dt[[i for i in range(len(k_list))]].values

	plt.figure(figsize=(4,2.5))
	prev = np.zeros(dt.shape[0])
	for f in range(dt.shape[1]):
	    p1 = plt.bar(range(dt.shape[0]), dt[:,f], 0.5, bottom=prev, label='C{}'.format(f+1))
	    prev = dt[:,:(f+1)].sum(axis=1)

	plt.legend(bbox_to_anchor=(1,0.5), loc='center left', ncol=1)
	plt.ylabel('Number of documents')
	plt.xticks(range(len(k_list)), ['{}'.format(i) for i in k_list])
	plt.xlabel('Total number of clusters (k)')
	plt.tight_layout()
	plt.savefig('Kmeans_clusterMembers{}.png'.format(dist))
	plt.close()



sc.stop()
