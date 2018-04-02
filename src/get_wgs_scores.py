#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:50:08 2017

@author: rrana
"""

import elasticsearch
from nltk.tokenize import RegexpTokenizer
from collections import Counter

es = elasticsearch.Elasticsearch([{'host': '10.30.150.102', 'port': 9200}])
my_table = 'smg360_response_363'     # Walgreens
block_size = 100
all_responses = {}
for start_count in xrange(0,500000,block_size):
	print start_count
	try:
	    q_set  = build_datefiltered_body(start = start_count, size = block_size)
	    res = es.search(index=my_table, size=block_size, body=q_set)
        for row in res['hits']['hits']:
        	d = {}
	    	osat = np.nan
	    	for msg in row['_source']['messages']:
				d[msg['fact']] = msg['text']
 		    for fact_dict in row['_source'][u'facts']:
		        try:
		       		if fact_dict['key'] == u'CouponNum':
	    	            uid = fact_dict[u'stringValue']
	        	    if fact_dict['key'] == u'R000001':
	            	    osat = fact_dict['numericValue']
  except:
	           	continue
	    d['osat'] = osat
 	    all_responses[uid] = d
	except:
	    print "except"
		time.sleep(10)
    	continue

named_responses = {}
res = {}
for name in candidate_names[0:5]:
	try:
    	q_set = build_body(search_term = name, row_count = row_count)
    	res = es.search(index=my_table, size=row_count, body=q_set)
	    for row in res['hits']['hits']:
	    	d={}
        	osat = np.nan
				for msg in row['_source']['messages']:
					d[msg['fact']] = msg['text']
				for fact_dict in row['_source'][u'facts']:
					try:
						if fact_dict['key'] == u'CouponNum':
							uid = fact_dict[u'stringValue']
						if fact_dict['key'] == u'R000001':
							osat = fact_dict['numericValue']
					except:
						continue
				if (osat == osat):           # a way to test for NaN
					d['osat'] = osat
					named_responses[uid] = d
	except:
		time.sleep(3)
		continue
        
df_walgreen_names = pd.DataFrame.from_dict(all_responses, orient = 'index')
df_walgreen_names.head(10)