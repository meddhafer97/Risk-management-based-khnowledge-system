# Risk-management-based-khnowledge-system
This repository contains a risk management based knowledge system based on two pdf files from where we've got the necessary knowledge we need, follow those two links to get the necessary dataset:<br/>
* [PMBOK-5th_Risk](./PMBOK%205th.pdf) 
* [PMIPracticeStandardforProjectRiskManagement](./PMIPracticeStandardforProjectRiskManagement.pdf) <br/> <br/>
you can test this project by running the django project folder (running the server) as you see bellow : <br/>
![testing](./testing.PNG) <br/>
## **Overview**

---

A knowledge-based system (KBS) is a form of artificial intelligence (AI) that aims to capture the knowledge of human experts to support decision-making, this project aims to get knowledge of risk management by using the most two popular books, then by adding some semantic web terms we'd be able to extract entities and relations which means that our system is able to get the most important things that it should keep in mind.<br/>
We did apply NLP techniques and now we have an ontology file that contains the knowledge extracted by the system (classes, subclasses, individuals, object properties, data properties and annotations) and you can open that file either with an IDE as visual code or by an ontology software as Protege.<br/>
After extracting the knowledge we have to use a matching algorithm to compare between the customer request and the entities existing on the ontology file in order to get an accurate answer.
## **Description of files

---

