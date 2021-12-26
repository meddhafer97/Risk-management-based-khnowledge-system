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
from django.shortcuts import render
from django.http import JsonResponse
import subprocess
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import UserForm
from .serializers import TaskSerializer

from .models import Task


# Create your views here.
gloveFile = "C:/Users/meddhafer/OneDrive/Bureau/Risk management based khnowledge system/glove.6B.50d.txt"
sp = spacy.load('en_core_web_sm')
annotations = pd.read_csv('C:/Users/meddhafer/OneDrive/Bureau/Risk management based khnowledge system/annotations.csv')
onto = get_ontology("C:/Users/meddhafer/OneDrive/Bureau/Risk management based khnowledge system/onto6.owl").load()
classes = list(onto.classes())
classes = [str(ont).split('.')[1] for ont in classes]
indivs = list(onto.individuals())
indivs = [str(ont).split('.')[1] for ont in indivs]


def fun(onto):
    dicto = {}
    classes = list(onto.classes())
    subs = [list(ont.subclasses()) for ont in classes if len(list(ont.subclasses())) > 0]
    indivs = list(onto.individuals())
    str_classes = [str(ont).split('.')[1] for ont in classes]
    str_indivs = [str(ont).split('.')[1] for ont in indivs]
    for i in range(len(classes)):
        my_dict = {'type': 'class', 'ontoform': classes[i], 'strform': str_classes[i]}
        for j in range(len(subs)):
            if my_dict['ontoform'] in subs[j]:
                my_dict['type'] = 'sub'
            else:
                continue
        dicto[i] = my_dict
    for i in range(len(classes), len(indivs) + len(classes)):
        my_dict = {'type': 'indiv', 'ontoform': indivs[i - len(classes)], 'strform': str_indivs[i - len(classes)]}
        dicto[i] = my_dict
    return dicto


def spell_it_right(myrequest):
    corrected = [str(TextBlob(word).correct()) for word in myrequest.split(' ')]
    myrequest = ' '.join(corrected)
    return myrequest


def fix_entities(entities):
    #     request2 = request.replace('_',' ')
    for t in range(len(entities)):
        t2 = entities[t].replace('_', ' ')
        entities[t] = t2
    #     entities.append(request2)
    return entities


def loadGloveModel(gloveFile):
    with open(gloveFile, encoding="utf8") as f:
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
    return (1 - scipy.spatial.distance.cosine(model[word1], model[word2]))


def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([model[word] for word in preprocess(s1)], axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)], axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return round((1 - cosine) * 100)


model = loadGloveModel(gloveFile)


def find_entity(myrequest, entities):
    score_list = []
    for ind in range(len(entities)):
        score_list.append(cosine_distance_wordembedding_method(myrequest, entities[ind]))
    score_list = list(enumerate(score_list))
    sorted_scores = sorted(score_list, key=lambda x: x[1], reverse=True)
    if sorted_scores[0][1] < 0.5:
        return None
    ind = sorted_scores[0][0]
    return entities[ind]


def annotation(individus, annotations):
    anno = ''
    for i in annotations.index:
        if annotations['about'][i] == individus:
            anno += annotations['description'][i]
            break
        else:
            continue
    return anno


def todo(entity, anno, onto, ontof):
    result = ''
    annot = ''
    if len(anno) != 0:
        result += 'Here is some information about ' + entity + ' :\n' + anno + '\n'
    subs = list(ontof.subclasses())
    if len(subs) > 0:
        return result
    result += 'In order to better understand the ' + entity + ", I'm going to give you several examples:\n"
    individis = onto.search(type=ontof)
    individis = [str(ont).split('.')[1] for ont in individis]
    if len(individis) == 0:
        return result
    for x in individis:
        annot = annotation(x, annotations)
        if len(annot) == 0:
            result += '• ' + x.replace('_', ' ') + '\n'
        else:
            result += '• ' + x.replace('_', ' ') + ' : ' + annot + '\n'
    return result


def final(entity, annotations, diction_entities, onto):
    entity2 = entity.replace(' ', '_')
    result = ''
    anno = ''
    typee = ''
    ontof = None
    anno = annotation(entity2, annotations)
    for k, v in diction_entities.items():
        if v['strform'] == entity2:
            typee = v['type']
            ontof = v['ontoform']
            break
    if typee == 'class':
        result = todo(entity, anno, onto, ontof)
    elif typee == 'sub':
        result = todo(entity, anno, onto, ontof)
    elif typee == 'indiv':
        if len(anno) != 0:
            result += 'Here is some information about ' + entity + ' :\n' + anno + '\n'
        else:
            result += "We're sorry, we don't have much information about " + entity
    return result

@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List': '/task-list/',
        'Detail View': '/task-detail/<str:pk>/',
        'Create': '/task-create/',
        'Update': '/task-update/<str:pk>/',
        'Delete': '/task-delete/<str:pk>/',
    }

    return Response(api_urls)


@api_view(['GET'])
def taskList(request):
    tasks = Task.objects.all().order_by('-id')
    serializer = TaskSerializer(tasks, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def taskDetail(request, pk):
    tasks = Task.objects.get(id=pk)
    serializer = TaskSerializer(tasks, many=False)
    return Response(serializer.data)


@api_view(['POST'])
def taskCreate(request):
    serializer = TaskSerializer(data=request.data)

    if serializer.is_valid():
        serializer.save()

    return Response(serializer.data)


@api_view(['POST'])
def taskUpdate(request, pk):
    task = Task.objects.get(id=pk)
    serializer = TaskSerializer(instance=task, data=request.data)

    if serializer.is_valid():
        serializer.save()

    return Response(serializer.data)


@api_view(['DELETE'])
def taskDelete(request, pk):
    task = Task.objects.get(id=pk)
    task.delete()

    return Response('Item succsesfully delete!')


def ontology(request):
    if request.method == "POST":
        myrequest= request.POST.get('n')

        print(myrequest)
    context={myrequest}
    return render(request,"risk.html",context)


def index(request):
    submitbutton = request.POST.get("submit")

    firstname = ''


    form = UserForm(request.POST or None)
    if form.is_valid():
        firstname = form.cleaned_data.get("first_name")

    context = {'form': form, 'firstname': firstname,
               'submitbutton': submitbutton}

    return render(request, 'm.html', context)

def home(request):

    myrequest=request.POST.get('requestrisk')
    myrequest=str(myrequest)
    diction_entities = fun(onto)

    myrequest = spell_it_right(myrequest)
    entities = []
    entities.extend(classes)
    entities.extend(indivs)
    entities = fix_entities(entities)
    entity = find_entity(myrequest, entities)
    if entity == None:
        result = 'We could not find any match for your request :('
    else:
        result = final(entity, annotations, diction_entities, onto)
    context={'myrequest': result}
    return render(request, 'home.html', context)
