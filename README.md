# Overview
This is an ongoing research project to develop structured knowledge from text data. Current supported features include retrieving concepts, retrieving outcomes, retrieving key terms, building a terminology from key terms, summarization, retrieval of main topics, dependencies between sections/chapters of text, and some visualization using plain graphs, hypergraphs, and interactive graphs. Evaluation is also supported using deepeval with ground truth values and retrieved contexts. Future features include knowledge graph creation, deeper visualization, and custom language model integration. 

The substance of the project is located in the src directory. The experiments directory, playground.ipynb, and testing.ipynb will not work without setting up an env file, so it is suggested to create your own work file. 

To use the main class, relationExtractor, you will need a MongoDB and OpenAI api token. 

Bulk testing is also available with run_test.sh and test_script.py, but you will need to edit them to use your own data. In the future they may become an automatic test, but for the foreseeable future it will stay this way. 

# Environment Setup
Create a virtual environment using python or conda:<br/>
Using Linux with python:
```
python3 -m venv env
```
Using Windows with python:
```
python -m venv env
```
Using conda:
```
conda create -n env python=3.10.12 anaconda
```
# Activation and package installation
## Activation
### Pip
Using Linux, activate python virtual environment using:<br/>
```
source env/bin/activate
```
Using Windows Powershell, activate python virtual environment using:
```
.\venv\Scripts\activate
```
or, if you are using Windows with a Unix-like CLI:
```
source env/Scripts/activate
```
### Conda
On Windows and Linux, activate conda virtual environment using:
```
conda activate env
```
## Package Installation
Using pip, install packages using:
```
pip install -r requirements.txt
```
Conda uses an environment.yaml file to specify dependencies instead of requirements.txt, so install pip first:
```
conda install pip
```
Then, install packages:
```
pip install -r requirements.txt
```
# data
Concepts for data structures course. Sorting file is only concepts for searching and sorting chapters

# experiments
Experimentation on three different textbooks (cloud computing, graduate DS, undergraduate DS)
These are based on the old architecture and will not currently run

# src
Source code
Extractor file - LLM_Relation_Extractor class file of the algorithms.
utils file - utilities used in main class (splitting textbook, creating vector store, etc)

# testing and playground files
playground - trying out different code and functions and seeing what works best 
testing - evaluation testing of the entire architecture using undergrad DS textbook

# notes / observations
The testing file is the main focus right now. I have observed several patterns when playing around with the validate function of the relationExtractor class. The evaluate() function provided by ragas is a little weird, and produces ambigious errors sometimes. Additionally, if llm and embedding arguments are given the metrics, the results significantly vary. 

For example, without an llm argument to LLMContextPrecisionWithReference, it's very high (1.0). Additionally, without an llm passed to LLMContextRecall, the performance is very poor (~0.20 - 0.30). However, if you add in arguments to these they essentially reverse. Precision becomes very poor (~0.0) and recall becomes very high (~1.0). 

Similarly, the same thing occurs with the FactualCorrectness metric. However, it tends to stay poor regardless of an argument. Without an llm it tends to be (0.0 - 0.06) and with an llm it becomes (0.2 - 0.3)

The only evaluations that are consistent are response relevancy (0.83 - 0.86), semantic similarity (0.80 - 0.83), and faithfulness (0.90 - 0.1).

Finally, evaluate() will sometimes produce these statements saying 'No statements were generated in the answer' and skips ahead. I'm not really sure what its doing, so I'm going to raise an issue on the github page and ask. My current theory of this is that if an llm is passed as an arg, it might say this and cause the metric to be unusually high.


TLDR: the evaluation results REALLY depend on the args passed to the metrics

UPDATE: Upon trying deepeval for evaluation, the answer relevancy and faithfulness scores are much more consistent, but deepeval produces an inconsistent JSONDecodeError within its metrics for some reason. Currently looking into the cause and fix 
