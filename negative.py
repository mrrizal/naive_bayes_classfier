import json
import re
from pprint import pprint

data = open('negative_tweets.json', 'r').read().split('\n')
f = open('negative_tweets.txt', 'w')
for i in data:
	try:
		text = json.loads(i)['text']
	except:
		continue

	if text != '':
		text = text+'	'+'negative'
		f.write(text.replace('\n', ''))
		f.write('\n')