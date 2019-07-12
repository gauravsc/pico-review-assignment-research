import csv
from collections import defaultdict
from collections import Counter

def extract_outcomes(path):
	outcome_dict = defaultdict(list)
	with open(path, 'r') as f:
		for line in f:
			row = line.split('\t')
			outcome_dict[row[0]].append(row[1])
	
	return outcome_dict	


def extract_intervention(path):
	intervention_dict = defaultdict(list)
	with open(path, 'r') as f:
		for line in f:
			row = line.split('\t')
			intervention_dict[row[0]].append(row[1])

	return intervention_dict


def extract_condition(path):
	condition_dict = defaultdict(list)
	with open(path, 'r') as f:
		for line in f:
			row = line.split('\t')
			condition_dict[row[0]].append(row[1])

	return condition_dict

def extarct_RID_to_CN(path):
	rid_to_cn = {}
	with open(path, 'r', errors='ignore') as f:
		for i, line in enumerate(f):
			if i == 0: 
				continue
			row = line.split('\t')
			rid_to_cn[row[0]] = row[10]
	return rid_to_cn

def remove_duplicates(data):
	for key in data.keys():
		data[key] = list(set(data[key]))

	return data


def count_per_review(rid_to_cn, condition_dict, intervention_dict, outcome_dict):
	review = defaultdict(list)
	rids = list(condition_dict.keys()) + list(intervention_dict.keys()) + list(outcome_dict.keys()) 
	
	for rid in rids:
		review[rid_to_cn[rid]] = review[rid_to_cn[rid]] + condition_dict[rid] + intervention_dict[rid] + outcome_dict[rid]

	for rid in rids:
		review[rid_to_cn[rid]] = Counter(review[rid_to_cn[rid]])

	for cn in review.keys():
		print (cn, review[cn])


def display(condition_dict, intervention_dict, outcome_dic):
	keys = list(condition_dict.keys()) + list(intervention_dict.keys()) + list(outcome_dict.keys()) 
	for key in keys:
		print (key, condition_dict[key], intervention_dict[key], outcome_dict[key])


if __name__ == '__main__':
	condition_dict = extract_condition('../Data/Condition.txt')
	intervention_dict = extract_intervention('../Data/InterventionType.txt')
	outcome_dict = extract_outcomes('../Data/OutcomeType.txt')
	rid_to_cn = extarct_RID_to_CN('../Data/CRSDLinkedData.txt')

	condition_dict = remove_duplicates(condition_dict)
	intervention_dict = remove_duplicates(intervention_dict)
	outcome_dict = remove_duplicates(outcome_dict)

	count_per_review(rid_to_cn, condition_dict, intervention_dict, outcome_dict)

	# display(condition_dict, intervention_dict, outcome_dict)