# Risk-management-khnowledge-based-system
This repository contains a risk management knowledge-based system based on two pdf files from where we've got the necessary knowledge we need, follow those two links to get the necessary dataset:<br/>
* [PMBOK-5th_Risk](./PMBOK%205th.pdf) 
* [PMIPracticeStandardforProjectRiskManagement](./PMIPracticeStandardforProjectRiskManagement.pdf) <br/> <br/>
you can test this project by running the Django project folder (running the server) as you see bellow : <br/>
![testing](./testing.PNG) <br/>
## **Overview**

---

A knowledge-based system (KBS) is a form of artificial intelligence (AI) that aims to capture the knowledge of human experts to support decision-making, this project aims to get knowledge of risk management by using the most two popular books, then by adding some semantic web terms we'd be able to extract entities and relations which means that our system can get the most important things that it should keep in mind.<br/>
We did apply NLP techniques and now we have an ontology file that contains the knowledge extracted by the system (classes, subclasses, individuals, object properties, data properties ,and annotations) and you can open that file either with an IDE as visual code or by an ontology software as Protege.<br/>
After extracting the knowledge we have to use a matching algorithm to compare between the customer request and the entities existing on the ontology file to get an accurate answer.
## **Description of files**

---

* DjangoProject1 : it's the deployment project (admin interface developed by Django, /api where we had used rest framework, and on /api/home you can find the principal interface as you see above.
* Text_preprocessing-6666.ipynb : it's the first notebbok you should analyze(it involves the first steps as preprocessing our data and applying NLP techniques)
* annotations.csv : contains the annotations we had figured out by applying some filtering and patterns
* glove_cosine.ipynb : contains a few pieces of codes that verify the accuracy of the model we're going to use
* onto6.owl : the ontology file that we've got
* owl.ipynb : we leverage the datasets we've got from the first notebook to get the ontology file with the aid of owlready2(ontology's API)
* owl2.py : this is the third step where we had used the ontology to extract our entities and we had preprocessed the request to compare it then with the entities by using a matching model (the result of running this python script would be the answer of a written request, you could enjoy testing your customized requests)
* relation.csv : a CSV file containing the relations we had found by using some patterns.
* subconcepts.csv : a CSV file containing relations between classes and subclasses.
* ## **Important notes**

---

* Please follow that [link](https://www.kaggle.com/watts2/glove6b50dtxt) to download the glove.6b.50d.txt file that is necessary to train the glove model (this model use the embedding term to give us a significative vector instead of a word then we'll use the cosine similarity to calculate the similarity between entities and the request entered by the customer).
* to better understand how to calculate text similarities, I suggest following that [link](https://medium.com/@adriensieg/text-similarities-da019229c894) that contains several methods well explained.
* Finally please read about semantic web and ontologies otherwise you'd feel lost.
