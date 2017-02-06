import re
from nltk.stem import porter
import requests
import json
import sys
from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials
from sklearn import tree
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

#######################################################################################
# Regex Entities
#######################################################################################

phone_regex = re.compile(r'[\d\+\(][\s\-\d\+\(\)\/\.]{5,}[\s\.,]*(?:x|ext|extension)?[\s\.,\d]*[\d\)]', re.I)
def extract_phones(text) :
  phones = phone_regex.findall(text)
  return phones

email_regex = re.compile(r'\b[\w\.\-!#$%&\'*\+\/=\?\^_`{\|}~]+@[\w\.\-]+\.[\w\-]+\b')
def extract_emails(text) :
  emails = email_regex.findall(text)
  return emails

with open('url_endings.txt','r') as f :
  url_endings = f.read().split('\n')
url_regex = re.compile(r"(?:https?\:\/\/)?[\w\-\.]+\.(?:"+"|".join(url_endings)+")[\w\-\._~:/\?#\[\]@!\$&%\'\(\)\*\+,;=]*")
def extract_urls(text) :
  urls = url_regex.findall(text)
  return urls


#######################################################################################
# Duckling Entities
#######################################################################################
                
# {'dim':'number', 'body':'5'} -> ('5','ENT/number')
def parse_ent_info(ent_info) :
  if ent_info['dim']=='time' :
    return (ent_info['body'].strip(), 'ENT/time')
  return None
    
# 'we went in March but now will go at 5pm' -> {'March':'ENT/month', '5pm':'ENT/time'}
def extract_entities(s) :

  # call my entity parsing service
  r = requests.get('http://localhost:3001', params={'s':s})
  response = json.loads(r.text)
  used_ents = set([])
  ents = [parse_ent_info(s) for s in response]

  # get rid of duplicates
  used = set([])
  for i in range(len(ents)) :
    if ents[i] == None or ents[i][0] in used :
      ents[i] = None
    else :
      used.add(ents[i][0])
  ents = filter(lambda x: x!=None, ents)

  return dict(ents)


#######################################################################################
# Preprocessing
#######################################################################################

# Add the feature 'SHORT' to the feature list if there are <= 6 words in the sentence
def add_short_feature(words, features) :
  if len(words) <= 6 :
    features.append('SHORT')


# Stem a word
stemmer = porter.PorterStemmer()
def stem(word) :
  return stemmer.stem(word)


# Load colloquialism resource
with open('colloquialism_lists/colloquialisms_english.txt', 'r') as f :
  lines = f.read().split('\n')
colloquialism_map = dict([line.split() for line in lines])

# words: ['y','r','u','going'] -> ['why','are','you','going']
def fix_colloquialisms(words) :
  new_words = list(words)
  for i in range(len(new_words)) :
    if new_words[i] in colloquialism_map :
      new_words[i] = colloquialism_map[new_words[i]]
  return new_words


# Spelling Correction
# Before calling this, start the server in the folder /spelling-server
def correct_sentence_spelling(s) :
  r = requests.get('http://localhost:3002', params={'s':s})
  response = r.text
  if 'TypeError' in response :
    print 'PROBLEM: Spellchecker returned TypeError for: '+s
    return s
  return response


###########################################################################
### Quickly find possible mispellings by method from http://emnlp2014.org/papers/pdf/EMNLP2014171.pdf
### 1. make a distinct letter -> prime number mapping
### 2. multiply primes for letters in word
### 3. find all entities with scores that are off by one or two prime factors (off by one or two letters)
### 4. run edit distance on this vastly reduced set of candidates to find if the incorrect letters are properly positioned
###########################################################################

# entname_set: ['star wars','star trek']
# returns ( {'a':2,'b':3,...}, {'star wars':1.232424e46,...}, [2,3,0.5,0.33,1.5,...] )
def make_mispelling_resources(entname_set) :

  # map letters to prime numbers
  primes_letters = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
  primes_numbers = [103,109,113,127,131,137,139,149,151,157]
  primes_all = primes_letters + primes_numbers + [163,167,173]
  primes_map = {' ':163,'-':167,'\'':173}
  for i in range(26) :
    primes_map[chr(ord('a')+i)] = primes_letters[i]
  for i in range(10) :
    primes_map[chr(ord('0')+i)] = primes_numbers[i]

  # list of factors that entity letter score can be off by for one or two errors
  possible_spelling_ratios = set( flatten_one_layer([[1.0*x*y,1.0*x/y,1.0*y/x,1.0/x/y] for x in primes_all for y in primes_all])
        + flatten_one_layer([[1.0*x,1.0/x] for x in primes_all]) )

  # map of spelling score to entity
  ent_spell_scores = {}
  for ent in entname_set :
    num_list = [primes_map[c] for c in ' '.join(ent)]
    if len(num_list)==0 or len(num_list)>40 :
      continue
    ent_spell_scores[float(reduce(op.mul,num_list))] = ent

  return (primes_map, ent_spell_scores, possible_spelling_ratios)

# source: http://stackoverflow.com/questions/2460177/edit-distance-in-python
def edit_distance(s1, s2):
  if len(s1) > len(s2):
    s1, s2 = s2, s1
  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2+1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    distances = distances_
  return distances[-1]

# return list of entities off by 1 or 2 letters from ent
def find_mispellings(ent, primes_map, ent_spell_scores, possible_spelling_ratios) :

  # check each of ~1000 values that this spelling can be off by
  # add to possibilities any value that is present in ent_spell_scores so corresponds to a known entity
  find_val = reduce(op.mul,[primes_map[c] for c in ' '.join(ent)])
  possibilities = []
  for ratio in possible_spelling_ratios :
    if find_val*ratio in ent_spell_scores :
      possibilities.append(ent_spell_scores[long(find_val*ratio)])

  # use expensive edit distance method on reduced list to account for letter order
  found_ents = []
  for poss in possibilities :
    if edit_distance(' '.join(poss),' '.join(ent))<=2 :
      found_ents.append((poss,ent))
  return found_ents


#######################################################################################
# Google Parse (part of speech tags, dependencies, root)
#######################################################################################

# pos: {'the':'det', 'run':'vb'}
# deps: [Dep,Dep,Dep]
# root: run
class Parse:
  def __init__(self,pos,deps,root) :
    self.pos = pos
    self.deps = deps
    self.root = root
        
  def __str__(self) :
    s='['
    s+=self.pos.__str__()+',\n'
    s+=self.deps.__str__()+',\n'
    s+=self.root.__str__()+'\n'
    return s

# [nsubj,dog,0,brown,2]
class Dep:
  def __init__(self,rel,w1,w1id,w2,w2id) :
    self.rel = rel
    self.w1 = w1
    self.w1id = w1id
    self.w2 = w2
    self.w2id = w2id
        
  def __str__(self) :
    return '[%s,%s,%i,%s,%i]' % (self.rel, self.w1, self.w1id, self.w2, self.w2id)
    
  def __repr__(self) :
    return self.__str__()

# returns the encoding type that matches Python's native strings.
def get_native_encoding_type():
  if sys.maxunicode == 65535:
    return 'UTF16'
  else:
    return 'UTF32'

# {'entities': [{'type':'a', 'name':'b'}]} -> {'a:'b'}
def format_entities(extracted_info) :
  return dict( [(ent_info['name'],[ent_info['type']]) for ent_info in extracted_info['entities']] )

# Call analyze_text for Google NLP API then formats response to Parse object
def format_parse(extracted_info) :
  tokens = extracted_info.get('tokens', [])

  # extract word ids and part of speech tags into words and pos dict
  words = {}
  pos = {}
  for i in range(len(tokens)) :
    token = tokens[i]
    word = token['text']['content']
    words[i] = word
    tag = token['partOfSpeech']['tag']
    pos[word] = tag.lower()

  # extract dependencies into list
  deps = []
  for i in range(len(tokens)) :
    token = tokens[i]
    dep = token['dependencyEdge']
    other_word = words[ dep['headTokenIndex'] ]
    deps.append( Dep(dep['label'].lower(), words[i], 0, other_word, 0) )

  return Parse(pos, deps, '')

# Call Google Natural Language syntax API, raises HTTPError is connection problem.
def get_dependency_parse(text):
  credentials = GoogleCredentials.get_application_default()
  scoped_credentials = credentials.create_scoped(['https://www.googleapis.com/auth/cloud-platform'])
  http = httplib2.Http()
  scoped_credentials.authorize(http)
  service = discovery.build(
      'language', 'v1beta1', http=http)
  body = {
      'document': {
          'type': 'PLAIN_TEXT',
          'content': text,
      },
      'features': {
          'extract_syntax': True,
      },
      'encodingType': get_native_encoding_type(),
  }
  request = service.documents().annotateText(body=body)
  extracted_info = request.execute()
  
  parse_info = format_parse(extracted_info)
  return parse_info


#######################################################################################
# Google Entities (people, locations)
#######################################################################################

# Call Google Natural Language syntax API, raises HTTPError if connection problem.
def get_proper_noun_entities(text):
  credentials = GoogleCredentials.get_application_default()
  scoped_credentials = credentials.create_scoped(['https://www.googleapis.com/auth/cloud-platform'])
  http = httplib2.Http()
  scoped_credentials.authorize(http)
  service = discovery.build(
      'language', 'v1beta1', http=http)
  body = {
      'document': {
          'type': 'PLAIN_TEXT',
          'content': text,
      },
      'features': {
          'extract_entities': True
      },
      'encodingType': get_native_encoding_type(),
  }
  request = service.documents().annotateText(body=body)
  extracted_info = request.execute()

  # parse ent format
  ent_info = {}
  for ent in extracted_info['entities'] :
    text = ent['mentions'][0]['text']['content'].lower()
    type = 'ENT/'+ent['type'].lower()
    ent_info[text] = [type]
  return ent_info



#######################################################################################
# Decision Tree
#######################################################################################

# assign a unique id from 0...n to each feature
# ex. {'and': 3, 'run': 4, 'i_run': 2, 'dog': 1, 'cat': 0, 'go': 5}
def map_feature_name2id(X_train) :
  feature_name2id = {}
  for row in X :
    for feat in row :
      if feat not in feature_name2id :
        feature_name2id[feat] = len(feature_name2id)
  return feature_name2id

# X: [ [first user input features], [2nd user input features], ...]
# Each input feature list should be represented like ['dog','SHORT','i_run','PHONE','cat']
# Output : sparse matrix format suitable for sklearn's decision tree model
def features2csr_matrix(X, feature_name2id) :
  
  # convert to csr format
  row_values = []
  col_values = []
  data_values = []
  for row_num in range(len(X)) :

    row_features = X[row_num]
    for feature in row_features :
            
      # never saw feature in training set
      if feature not in feature_name2id :
        continue
      feature_id = feature_name2id[feature]
      row_values.append(row_num)
      col_values.append(feature_id)
      data_values.append(1)
            
  return csr_matrix((data_values, (row_values, col_values)), shape=(len(X),len(feature_name2id)))


# Sample inputs
# X: [['cat','dog','i_run'],['and','run','go'],['cat','dog'],['and','run','go'],['cat','dog']]
# y: [0,1,0,1,0]
# max_leaf_nodes: 60
# Returns classifier
def train_classifier(X, y, max_leaf_nodes) :

  # split the dataset into train and test splits
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)
  
  # assign a unique id from 0...n to each feature
  # ex. {'and': 3, 'run': 4, 'i_run': 2, 'dog': 1, 'cat': 0, 'go': 5}
  feature_name2id = map_feature_name2id(X_train)

  # convert the input data into sparse format
  X_train_sparse = features2csr_matrix(X_train, feature_name2id)
  X_test_sparse = features2csr_matrix(X_test, feature_name2id)

  # Fit decision tree model
  clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes) 
  clf = clf.fit(X_train_sparse, y_train)
  
  # Show test performance
  print 'Test accuracy: ' + str(clf.score(X_test_sparse, y_test))

  return clf

# features: ['cat','dog','i_run']
# returns: class probabilities ex. [0.5,0.3,0.2]
def predict_classification(clf, features) :
  sparse_features = features2csr_matrix([features], feature_name2id)
  return clf.predict_proba(sparse_features)[0]




