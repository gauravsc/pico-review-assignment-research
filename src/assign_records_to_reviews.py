import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support


data_CRSD = pd.read_csv('../data/CRSDLinkedData.txt', encoding = "ISO-8859-1", error_bad_lines=False, delimiter='\t', usecols = ['RecordId', 'cn'])
data_outcome = pd.read_csv('../data/OutcomeType.txt', encoding = "ISO-8859-1", error_bad_lines=False, delimiter='\t', usecols=['RecordId', 'Outcome'], header=None, names=['RecordId', 'Outcome'])
data_intervention = pd.read_csv('../data/InterventionType.txt', encoding = "ISO-8859-1", error_bad_lines=False, delimiter='\t', usecols=['RecordId', 'Intervention'], header=None, names=['RecordId', 'Intervention'])
data_population = pd.read_csv('../data/Condition.txt', encoding = "ISO-8859-1", error_bad_lines=False, delimiter='\t', usecols=['RecordId', 'Population'], header=None, names=['RecordId', 'Population'])


# remove NaN values from dataframes
data_CRSD = data_CRSD.dropna()
data_population = data_population.dropna()
data_intervention = data_intervention.dropna()
data_outcome = data_outcome.dropna()

# merge all the data frames on the RecordId column
merged_df = data_CRSD.join(data_population.set_index('RecordId'), on='RecordId', how='left')
merged_df = merged_df.join(data_intervention.set_index('RecordId'), on='RecordId', how='left')
merged_df = merged_df.join(data_outcome.set_index('RecordId'), on='RecordId', how='left')

# create two dataframes, one that can correspond to multi-label assignment matrix and the other to inpute features
df1 = merged_df[['RecordId', 'cn']]
df2 = merged_df[['RecordId', 'Population', 'Intervention', 'Outcome']]

# convert categorical features to one-hot encoding 
df1 = pd.get_dummies(df1, columns=['cn'])
df2 = pd.get_dummies(df2, columns=['Population', 'Intervention', 'Outcome'], prefix=['$Population$', '$Intervention$', '$Outcome$'])

# make sure that the one-hot remains valid even after summing
df1 = df1.groupby(['RecordId']).sum()
df1[df1>1] = 1

df2 = df2.groupby(['RecordId']).sum()
df2[df2>1] = 1

# convert to matrices to be trained using scikit
X = df2[df2.columns[1:]].values
Y = df1[df1.columns[1:]].values

num_records = np.sum(Y, axis=0)
Y = Y[:, num_records>4]

# classifier initialized
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial')

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; score_labelwise = []
for i in range(Y.shape[1]):
	y = Y[:,i]
	skf = StratifiedKFold(n_splits=2)
	skf.get_n_splits(X, y)
	splits = list(skf.split(X, y))

	score_thresholdwise = []
	for threshold in thresholds:
		scores = []
		for train_index, test_index in splits:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			clf.fit(X_train, y_train)
			pred = clf.predict_proba(X_test)[:,1]
			y_test_pred = np.array(pred)
			y_test_pred[y_test_pred>=threshold] = 1
			y_test_pred[y_test_pred<threshold] = 0

			prec, rec, fscore, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
			print (prec, rec, fscore)
			scores.append((prec, rec, fscore))
		
		prec = np.mean([score[0] for score in scores])
		rec = np.mean([score[1] for score in scores])
		f1 = np.mean([score[2] for score in scores])

		score_thresholdwise.append((prec, rec, f1))
		# print (score_thresholdwise)

	index = np.argmax([score[2] for score in score_thresholdwise])
	print ("Best scores of ",  score_thresholdwise[index]," for label: ", i, " at threshold: ", thresholds[index])
	score_labelwise.append(score_thresholdwise[index]) 



best_score = [[np.mean([score[0] for score in score_labelwise]), np.mean([score[1] for score in score_labelwise]), np.mean([score[2] for score in score_labelwise])]]


print (best_score)

