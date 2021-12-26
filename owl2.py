import pandas as pd
import sys
import numpy as np
from owlready2 import *
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk import ngrams 
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from nltk.corpus import stopwords
gloveFile = "glove.6B.50d.txt"
sp = spacy.load('en_core_web_sm')
annotations = pd.read_csv('annotations.csv')
onto = get_ontology("file://C:/Users/meddhafer/notebook/onto6.owl").load()
classes = list(onto.classes())
classes = [str(ont).split('.')[1] for ont in classes]
indivs = list(onto.individuals())
indivs = [str(ont).split('.')[1] for ont in indivs]
entities = []
entities.extend(classes)
entities.extend(indivs)
def fun (onto):
    dicto = {}
    classes = list(onto.classes())
    subs = [list(ont.subclasses()) for ont in classes if len(list(ont.subclasses()))>0]
    indivs = list(onto.individuals())
    str_classes = [str(ont).split('.')[1] for ont in classes] 
    str_indivs = [str(ont).split('.')[1] for ont in indivs]
    for i in range(len(classes)):
        my_dict = {'type':'class','ontoform':classes[i],'strform':str_classes[i]}
        for j in range(len(subs)):
            if my_dict['ontoform'] in subs[j]:
                my_dict['type'] = 'sub'
            else:
                    continue
        dicto[i] = my_dict
    for i in range(len(classes),len(indivs)+len(classes)):
        my_dict = {'type':'indiv','ontoform':indivs[i-len(classes)],'strform':str_indivs[i-len(classes)]}
        dicto[i] = my_dict
    return dicto
def spell_it_right (request):
    corrected = [str(TextBlob(word).correct()) for word in request.split(' ')]
    request = ' '.join(corrected)
    return request
# def find_terms (ch):
#     doc = sp(ch)
#     dicto = {}
#     noun = []
#     compounds = []
#     result = ''
#     for i in range(1,len(ch.split(' '))):
#         ngram = ngrams(request.split(' '), n=i)
#         dicto[i] = [tup for tup in ngram]
#     for k,v in dicto.items():
#         if k==1:
#             continue
#         for t in dicto[k]:
#             t = list(t)
#             sen = sp(' '.join(t))
#             for token in sen:
#                 if token.pos_!='NOUN':
#                     break
#                 elif token.dep_=='compound' or token.pos_=='NOUN':
#                     noun.append(token.text)
#                 elif token.pos_=='NOUN' and len(noun)>0:
#                     noun.append(token.text)
#                 else:
#                     break
#             if len(noun)==len(t):
#                 compounds.append(' '.join(noun))
#     for t in range(len(compounds)):
#         for text2 in compounds:
#             if compounds[t] in text2:
#                 compounds[t] = text2
#     compounds = set(compounds)
#     for comp in compounds:
#         ch = ch[:ch.find(comp)]+'_'.join(comp.split(' '))+ch[ch.find(comp)+len(comp):]
#     return ch
# def formatt (request):
#     l = ['identify','perform','plan']
#     words = request.split(' ')
#     for w in range(len(words)):
#         if words[w] in l:
#             if w+1<len(words) and any([words[w+1].startswith(ch) for ch in ['risk','risks','quantitative','qualitative']]):
#                 if w+1==len(words)-1:
#                     request = ' '.join(words[:w])+' '+words[w]+'_'+words[w+1]
#                 else:
#                     request = ' '.join(words[:w])+' '+words[w]+'_'+words[w+1]+' '+words[w+2:]
#     return request
def fix_entities(entities):
#     request2 = request.replace('_',' ')
    for t in range(len(entities)):
        t2 = entities[t].replace('_',' ')
        entities[t] = t2
#     entities.append(request2)
    return entities
def loadGloveModel(gloveFile):
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model
def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words
def cosine_distance_between_two_words(word1, word2):
    import scipy
    return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))
# def calculate_heat_matrix_for_two_sentences(s1,s2):
#     s1 = preprocess(s1)
#     s2 = preprocess(s2)
#     result_list = [[cosine_distance_between_two_words(word1, word2) for word2 in s2] for word1 in s1]
#     result_df = pd.DataFrame(result_list)
#     result_df.columns = s2
#     result_df.index = s1
#     return result_df

def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return round((1-cosine)*100)
    
# df = calculate_heat_matrix_for_two_sentences(s1,s2)
# print(cosine_distance_wordembedding_method(s1, s2))
model = loadGloveModel(gloveFile)
def find_entity (request,entities):
    score_list = []
    for ind in range(len(entities)):
        score_list.append(cosine_distance_wordembedding_method(request, entities[ind]))
    score_list = list(enumerate(score_list))
    sorted_scores = sorted(score_list,key=lambda x:x[1],reverse=True)
    if sorted_scores[0][1]<0.5:
        return None
    ind = sorted_scores[0][0]
    return entities[ind]
def annotation (individus,annotations):
    anno = ''
    for i in annotations.index:
        if annotations['about'][i] == individus:
            anno+=annotations['description'][i]
            break
        else:
            continue
    return anno
def todo (entity,anno,onto,ontof):
    result = ''
    annot = ''
    if len(anno)!=0:
        result+='Here is some information about '+entity+' :\n'+anno+'\n'
    subs = list(ontof.subclasses())
    if len(subs)>0:
        return result
    result+= 'In order to better understand the '+entity+", I'm going to give you several examples:\n"
    individis = onto.search(type=ontof)
    individis = [str(ont).split('.')[1] for ont in individis]
    if len(individis)==0:
        return result
    for x in individis:
        annot = annotation(x,annotations)
        if len(annot)==0:
                result+='• '+x.replace('_',' ')+'\n'
        else:
            result+='• '+x.replace('_',' ')+' : '+annot+'\n'
    return result
def final (entity,annotations,diction_entities,onto):
    entity2 = entity.replace(' ','_')
    result = ''
    anno = ''
    typee = ''
    ontof = None
    anno = annotation(entity2,annotations)
    for k,v in diction_entities.items():
        if v['strform'] == entity2:
            typee = v['type']
            ontof = v['ontoform']
            break
    if typee == 'class':
        result = todo (entity,anno,onto,ontof)
    elif typee == 'sub':
        result = todo (entity,anno,onto,ontof)
    elif typee == 'indiv':
        if len(anno)!=0:
            result+='Here is some information about '+entity+' :\n'+anno+'\n'
        else:
            result+="We're sorry, we don't have much information about "+entity
    return result
diction_entities = fun(onto)
request = 'who is me'
request = spell_it_right(request)
# request = find_terms(request)
# request = formatt(request)
entities = fix_entities(entities)
entity = find_entity(request,entities)
if entity == None:
    result = 'We could not find any match for your request :('
else:
    result = final (entity,annotations,diction_entities,onto)
print(result)