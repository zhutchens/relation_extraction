import graphviz
import os
import re
from IPython.display import display, Image
from pyvis.network import Network
import hypernetx as hnx
import matplotlib.pyplot as plt
import plotly.express as px 
import graphlib
from src.retrieval import RetrievalSystem
from deepeval.test_case import LLMTestCase
from src.utils import create_concept_graph_structure, clean, process_pair
from concurrent.futures import ThreadPoolExecutor
from deepeval.models import DeepEvalBaseLLM
from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
import validators
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


# NOTE: Currently one main issue, the function that finds the associations between chapters seems to be broken. I think its the algorithm thats wrong. It also takes 10+ minutes to run
class RAGKGGenerator:
    '''
    
    '''
    def __init__(self, 
                chapters: list[str] | list[list[str]], 
                chunk_size: int,
                chunk_overlap: int,
                llm: DeepEvalBaseLLM, 
                textbooks: list[str] | str,
                syllabus: str = None,
                st_model: str = 'msmarco-distilbert-base-tas-b',
                ):
        '''
        Constructor to create a large language model relation extractor class. 

        Args:
            chapters (list[str] or list[list[str]]): list of chapter/section names in link
            chunk_size (int): number of characters in a chunk of content
            chunk_overlap (int): number characters to overlap between content chunks
            llm (DeepEvalBaseLLM): LLM to use. Use one the preconfigured language model classes from src.llms and pass in your model name (optional)
            textbooks (list[str] | str): textbooks to use 
            syllabus (str, default None): syllabus to use
            st_model (str, default msmarco-distilbert-base-tas-b'): sentence transformer to use 
        '''
        if not isinstance(textbooks, list) and not isinstance(textbooks, str): 
            raise ValueError(f'document must be of type list or str, got type {type(textbooks)}')

        self.textbooks = textbooks

        if os.path.exists(syllabus):
            self.syllabus = UnstructuredFileLoader(syllabus).load()
        elif validators.url(syllabus):
            self.syllabus = UnstructuredURLLoader([syllabus]).load()
        else:
            self.syllabus = syllabus

        self.chapters = chapters
         
        self.embedding_model = st_model

        # this seems like bad practice but whatever lol
        self.terminology, self.kg, self.main_topics, self.dependencies, self.main_topic_relationships, self.summarization, self.build_terminology_embeddings = None, None, None, None, None, None, None

        self.concepts, self.outcomes, self.key_terms = [], [], []
        self.retrieved_concept_context, self.retrieved_outcome_context, self.retrieved_term_context = {}, {}, {}

        self.llm = llm 

        self.retriever = RetrievalSystem(self.textbooks, chunk_size, chunk_overlap, st_model)

    
    def syllabus_pipline(self) -> tuple:
        '''
        This pipeline extracts main learning topics and objectives from syllabus, clusters them with relevant terms, and builds a knowledge graph 

        Args:
            None

        Returns:
            Network: pyvis network 
        '''
        if self.syllabus is None:
            print('syllabus attribute cannot be None')

        if self.textbooks is None:
            print('textbooks attribute cannot be None')

        # task a
        self.identify_main_topics() 
        
        objective_prompt = f'''
                Identify the learning objectives from this syllabus: {self.syllabus}

                Format:
                objective_1::objective_2::objective_3...objective_n
                '''
            # for v in topic_clusters.values():
            #     for next_v in topic_clusters.values():
            #         if v != next_v:
            #             response = self.llm.generate(f'Identify if these values: {v} have an is-a relationship with these values: {next_v}. You should only respond with True or False')
            #             if response == 'True':
            #                 topic_relationships[topic].append((v, next_v))
        # task b
        syllabus_objectives = self.llm.generate(objective_prompt) 

        topic_objectives = []
        topic_relationships = {}
        topic_key_points = []
        clusters = []
        self.pcas = []

        for topic in self.main_topics: 
            # task c. i
            # print('topic:', topic)
            topic_context = self.retriever.pipeline(topic, self.llm) 
            key_points = self.llm.generate(f'''Extract learning key points from this context: {topic_context}. 
                                               
                                               Response Format:
                                               point1::point2::point3::...pointN
                                            ''').split('::')

            topic_key_points.append(key_points)

            # task c. ii
            relevant_terms = self.llm.generate(f''' 
                                                Extract semantically similar terms from these key points: {key_points}
                                                Relevant context: {topic_context}

                                                Format:
                                                term_1::term_2::term_3::...term_n
                                                ''').split('::') # task c. ii

            # task c. iii 
            topic_clusters = self.build_terminology(relevant_terms)
            clusters.append(topic_clusters)

            # print('cluster:', topic_cluster)
            # print('-' * 20)

            # task c. iv (i feel like i did this is incorrect)
            topic_relationships[topic] = []

            for v in topic_clusters.values():
                for next_v in topic_clusters.values():
                    if v != next_v:
                        response = self.llm.generate(f'Identify if these values: {v} have an is-a relationship with these values: {next_v}. You should only respond with True or False')
                        if response == 'True':
                            topic_relationships[topic].append((v, next_v))

            # task c. v
            topic_objectives.append(self.llm.generate(f'Extract relevant learning objectives from {topic}'))

        # task d
        # print('part d')
        # for first_topic, second_topic in topic_relationships:
        #     print('first:', first_topic, 'second:', second_topic)
        #     semantic_relationships = self.llm.generate(f'List semantic relationships beyond the is-a relationship between this pair {first_topic}, {second_topic}:')
        #     print(semantic_relationships)
        #     print('-' * 20)

        self.clusters = clusters # for cluster visualization

        self.topic_key_points = topic_key_points
        self.topic_objectives = topic_objectives
        self.identify_dependencies(dependencies_from = 'topics')
        self.kg = self.build_knowledge_graph(self.dependencies, self.topic_key_points, self.topic_objectives)

        return self.kg

    
    def from_question(self, question: str) -> tuple[list[str], list[str]]:
        '''
        Retrieves learning concepts and outcomes this question addresses

        Args:
            question (str): question to assess

        Returns:
            tuple: concepts and outcomes
        '''
        if not self.concepts:
            self.identify_concepts(10)
        if not self.outcomes:
            self.identify_outcomes(10)

        concept_prompt = f'''
                        Given this question: {question}, list relevant learning concepts from this list of learning concepts that it assess. 

                        Learning concepts: {self.concepts}
                        '''

        outcome_prompt = f'''
                        Given this question: {question}, list relevant learning outcomes from this list of learning outcomes that it assess. 

                        Learning outcomes: {self.outcomes}
                        '''

        outcomes = self.llm.generate(outcome_prompt).split('::')
        concepts = self.llm.generate(concept_prompt).split('::')
        return (concepts, outcomes)


    def get_misconceptions(self, question: str, answer: str) -> str:
        '''
        Retrieves student misconceptions given a question and answer

        Args:
            question (str): question given to student
            answer (str): student answer

        Returns:
            str: misconceptions 
        '''
        return self.llm.generate(f'Given this question: {question}, and this student response: {answer}, provide misconceptions the student may have about the question and topic.')

    
    def visualize_hierarchy(self, terminology: dict[str, list[str]], visual_type: str = 'sunburst'):
        '''
        Visualizes hierarchy from build_terminology using sunburst chart or cluster map

        Args:
            terminology (dict[str, list[str]]): clusters and their terms 
            cluster_predictions (list[int]): list of cluster predictions 
            visualize_type (str, default sunburst): type of visualization to use (sunburst/cluster map)

        Returns:
            None
        '''
        if visual_type.lower() == 'sunburst':
            # df = DataFrame(terminology)

            # columns = ['type']
            # for i in range(len(terminology[list(terminology.keys())[0]])):
            #     columns.append(f'term {i}')
            # df.columns = columns
            data = {
                'terms': [word for l in terminology.values() for word in l],
                'parent': [],
            }
            fig = px.sunburst()
            fig.show()

        else:   
            # show dendrogram and clusters for agglomerative clustering
            for pca, word in zip(self.pcas, self.main_topics):
                fig, ax = plt.subplots(1, 2, figsize = (12, 6))
                ax[0].scatter(pca[:, 0], pca[:, 1], c = self.ac.fit_predict(pca), cmap = 'viridis')
                ax[0].set_title(f'Clusters for main topic: {word}')

                matrix = linkage(pca, method = 'ward')
                dendrogram(matrix, ax = ax[1])
                ax[1].set_title(f'Dendrogram for main topic: {word}')

                plt.tight_layout()
                plt.show()


            
    def build_terminology(self, build_using: list[str]) -> dict[str, list[str]]:
        '''
        Builds hierarchy of concepts or terms using hierarchical clustering

        Args:
            build_using (list[str]): strings to build clusters with 

        Returns:
            tuple[dict[str, list[str]], list[int]]: clusters, cluster number predictions 
        '''
        scores = []
        pca = PCA()

        # need to use this in visualize_hierarchy, dont want to add it as a param (something else to return!)
        self.build_terminology_embeddings = pca.fit_transform(self.retriever.embedder.encode(build_using))
        self.pcas.append(self.build_terminology_embeddings)

        # finding optimal number of clusters using silhouette scores
        for i in range(2, 6):
            ac = AgglomerativeClustering(i).fit(self.build_terminology_embeddings)
            scores.append(silhouette_score(self.build_terminology_embeddings, ac.labels_))
        
        optimal_k = scores.index(max(scores)) + 1 # account for indices starting at 0 

        self.ac = AgglomerativeClustering(optimal_k)
        preds = ac.fit_predict(self.build_terminology_embeddings).tolist()

        topic_clusters = {}
        for i in range(1, optimal_k + 1):
            # choose type based on term closest to center of cluster
            # ex. topic_clusters[term closest to center] = []
            topic_clusters[f'Cluster {i}'] = []

            for j in range(len(preds)):
                if preds[j] == i:
                    topic_clusters[f'Cluster {i}'].append(build_using[j])

        # return topic_clusters, preds 
        return topic_clusters
        

    def textbook_pipline(self) -> tuple[list[list[str]], list[list[str]], dict[str, list[str]]]:
        '''
        In this pipeline, learning concepts and outcomes are extracted from each chapter and evaluated. Chapter dependencies are also created using the learning concepts

        Plots will be generated for chapter dependencies, concept hierachy, and knowledge graph. Concept hierarchies and knowledge graphs will be in ./visualizations/

        Args:
            None

        Returns:
            tuple[list[list[str]], list[list[str]], dict[str, list[str]]]: concepts, outcomes, dependencies 
        '''
        if self.textbooks is None:
            print('textbooks attribute cannot be None')

        self.identify_concepts(10)
        self.identify_outcomes(10)

        self.identify_dependencies()
        # self.build_terminology()

        self.kg = self.build_knowledge_graph(self.dependencies, self.concepts, self.outcomes)

        return self.kg

        
    def identify_key_terms(self, n_terms: int, input_type: str = 'chapter', chapter_name: str = None, concepts: list[list[str]] = None) -> list[str] | dict[str, list[str]]:
        '''
        Identify the key terms for a chapter or group of concepts
        
        Args:
            n_terms (int): number of key terms to use
            input_type (str, default chapter): if chapter, get key terms of a chapter. if concepts, get key terms for a list of concepts
            chapter_name (str, default None): name of chapter to get key terms for 
            concepts (list[str], default None): list of concepts to get key terms for

        Returns:
            list[str]: if getting key terms for a chapter\n
            dict[str, list[str]]: if getting key terms from a list of concepts
        '''
        # if not isinstance(n_terms, int):
        #     raise TypeError(f'n_terms must be int type, got {type(n_terms)}')

        # if concepts is None and chapter_name is None:
        #     raise ValueError(f'chapter_name and concepts cannot both be None')
        
        # if input_type == 'chapter':
        #     prompt = f'''
        #             Identify {n_terms} key terms from Chapter {chapter_name} in descending order of significance, listing the most significant terms first. The textbook is available here: {self.document}.
        #             For each term, provide the following:
        #             - Confidence Interval (CI),
        #             - Corresponding Statistical Significance (p-value),
        #             - A brief explanation of the term's relevance and importance,
        #             Format:
        #             term :: confidence interval :: p-value :: explanation
        #             '''

        #     terms = self.llm.invoke(prompt).content 

        #     return [string for string in terms.split('\n') if string != '']

        # # step 2
        # elif input_type == 'concepts':
        #     concept_terms = {}
        #     for concept in concepts:
        #         concept = ' '.join(concept)

        #         prompt = f'''
        #                 Identify {n_terms} key terms for the following concept: {concept}. 
        #                 Provide your answer in the following format:

        #                 Concept 1,
        #                 Concept 2,
        #                 Concept 3, 
        #                 ...
        #                 Concept n
        #                 '''

        #         words = self.llm.invoke(prompt).content
        #         concept_terms[concept] = [string for string in words.split('\n') if string != '']

        #     return concept_terms

        # else:
        #     raise ValueError(f'input_type value must be chapter or concepts, got {input_type}')
        if chapter_name is not None:
            return self.llm.generate(f'Identify {n_terms} key terms for {chapter_name}').split('\n')

        for chapter in self.chapters:
            relevant_docs = [doc for doc in self.retriever.pipeline(chapter, self.llm)]
            self.retrieved_term_context[chapter] = relevant_docs

            prompt = f'''
                    Identify {n_terms} key terms for chapter {chapter}. 
                    Relevant documents can be found here: {relevant_docs}

                    Provide your answer in the following format:

                    key term 1,
                    key term 2,
                    key term 3, 
                    ...
                    key term n
                    '''

            words = self.llm.generate(prompt)
            self.key_terms.append([string for string in words.split('\n') if string != ''])

        return self.key_terms, self.retrieved_term_context


    def summarize(self) -> str:
        '''
        Returns summary of object web link

        Args:
            None

        Returns:
            str: summary of textbook
        '''
        self.summarization = self.llm.generate(f"Please summarize this content: {self.document}")
        return self.summarization

    
    def objectives_from_syllabus(self):
        '''
        Identify objectives from syllabus

        Args:
            None

        Returns:
            list[str]: list of objectives from syllabus
        '''
        prompt = f'''
        Identify the main learning objectives from the provided syllabus.

        Syllabus: {self.syllabus}

        Response Format:
        objective_1::objective_2::objective_3::...objective_n
        '''
        return self.llm.generate(prompt).split('::')


    def identify_main_topics(self) -> list[str]:
        '''
        Identify the main topics from the syllabus

        Args:
            None
        
        Returns:
            list[str]: list of main topics 
        '''
        prompt = f'''
                    Please identify the main learning topics from this syllabus: {self.syllabus}

                    Format:
                    first main topic::second main topic::...nth math topic

                    Example:
                    Trees::Grass::Leaves::Photosynthesis
                  '''
        main_topics = self.llm.generate(prompt).split('::')
        self.main_topics = main_topics
        return self.main_topics
    
    
    def identify_main_topic_relations(self) -> dict[str, list[str]]:
        '''
        Identify the relationships between the main topics of the textbook

        Args:
            main_topics_list (list[str]): A list of main topics from the textbook. Can be automatically created using the identify_main_topics() function
        
        Returns:
            dict[str, list[str]]: relationships between main topics as adjacency list
        '''
        topic_relations = create_concept_graph_structure(self.main_topics)

        relation = ''
        for i in range(len(self.main_topics)):
            for j in range(len(self.main_topics)):
                if i != j:
                    relation = self.llm.generate(f"Is there a relationship between this topic: {self.main_topics[i]}, and this topic: {self.main_topics[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split(',')[0].strip() != 'No':
                        topic_relations[self.main_topics[i]].append(self.main_topics[j])

        self.main_topic_relationships = topic_relations
        return topic_relations

    
    def identify_outcomes(self, num_outcomes: int, concepts: list[list[str]] = None, concepts_keyTerms: dict[str, list[str]] = None) -> tuple[dict[str, str], list[list[str]]]:
        '''
        Identify main learning outcomes within the class provided link. If concepts and concepts_KeyTerms are both None, get outcomes from web link content

        Args:
            num_concepts (int): number of outcomes to generate
            concepts (list[list[str]]): list of concepts from identify_concepts()
            concepts_keyTerms (dict[str, list[str]]): dictionary of concepts and key_terms from identify_key_terms
            
        Returns:
            list[list[str]]: list of learning outcomes for each concept (or for each chapter if both args are None)\n
            dict[tuple[str, str], list[str]]: main concept and terms as key, list of outcomes as value
        '''
        # if concepts is None and concepts_keyTerms is None:

        for name in self.chapters:
            relevant_docs = [doc for doc in self.retriever.pipeline(name, self.llm)]
        
            # # single shot prompt 
            single_prompt = f'''
                             Given the following context, please identify the {num_outcomes} most important learning outcomes related to the chapter on {name}. 
                             Your response should directly reference key outcomes from {name} using the context provided.

                             Context: {relevant_docs}
                             
                             Additionally, use the following format for your response:
                             Outcome 1,
                             Outcome 2,
                             Outcome 3,
                             Outcome 4,
                             .
                             .
                             .
                             Outcome n

                             Example Output:
                             Demonstrate an understanding of polymorphism,
                             Demonstrate an ability to implement inheritance,
                             Demonstrate an ability to write classes,
                             Demonstrate an understanding of objects,
                             Explain core concepts behind encapsulation
                             '''

            self.retrieved_outcome_context[name] = relevant_docs

            response = self.llm.generate(single_prompt)

            self.outcomes.append([outcome for outcome in response.split('\n') if outcome != ''])

        return self.outcomes, self.retrieved_outcome_context
        
        # elif concepts is not None and concepts_keyTerms is None:
        #     for concept in concepts:
        #         concept = ' '.join(concept)
        #         response = self.llm.invoke(f'Identify the main learning outcomes from these concepts: {concept}').content
        #         outcome_list.append([outcome for outcome in response.split('\n') if outcome != ''])
            
        #     return outcome_list

        # elif concepts is None and concepts_keyTerms is not None:
        #     outcome_list = {}
        #     for k in concepts_keyTerms.keys():
        #         terms = ' '.join(concepts_keyTerms[k])
        #         response = self.llm.invoke(f'Identify the main learning outcomes given this concept {k} and these key terms {terms}').content

        #         outcome_list[(k, terms)] = [outcome for outcome in response.split('\n') if outcome != '']

        #     return outcome_list
        # else:
        #     raise ValueError(f'concepts and key_terms cannot both have a value. one of them must be None')


    def identify_concepts(self, num_concepts: int) -> tuple[dict[str, str], list[list[str]]]:
        '''
        Identify the main learning concepts within the class provided link

        Args:
            num_concepts (int): number of concepts to get per chapter
        
        Returns:
            tuple[dict[str, str], list[list[str]]]: retrieved contexts and list of concepts for each chapter
        '''
        for name in self.chapters:
            relevant_docs = [doc for doc in self.retriever.pipeline(name, self.llm)]
            
            # # single shot prompt 
            single_prompt = f'''
                             Given the following context, please identify the {num_concepts} most important learning concepts related to the chapter on {name}. 
                             Your response should directly reference key concepts from {name} using the context provided.

                             Context: {relevant_docs}
                             
                             Additionally, use the following format for your response:
                             Concept 1,
                             Concept 2,
                             Concept 3,
                             Concept 4,
                             .
                             .
                             .
                             Concept n

                             Example Output:
                             Polymorphism,
                             Inheritance,
                             Classes,
                             Objects,
                             Encapsulation
                             '''

            self.retrieved_concept_context[name] = relevant_docs

            current_concept = self.llm.generate(single_prompt)
            self.concepts.append([concept for concept in current_concept.split('\n') if concept != ''])

        return self.concepts, self.retrieved_concept_context

    
    # def get_assocations(self, first: list[list[str]] | list[str], second: list[list[str]] | list[str]) -> list[str]:
    #     '''
    #     Get the assocations between the first list and second list

    #     Args:
    #         first (list[list[str]] | list[str]): first list (concepts, key terms, outcomes, etc)
    #         second (list[list[str]] | list[str]): second list (concepts, key terms, outcomes, etc)

    #     Returns:
    #         list[tuple[str, str, str]]
    #     '''
    #     associations = []

    #     for f in first:
    #         if isinstance(f, list):
    #             f = ' '.join(f)

    #         for s in second:
    #             if isinstance(s, list):
    #                 s = ' '.join(s)

    #             prompt = f'''
    #                     Identify if there is some assocation between {f} and {s}. If there is an assocation, your response should ONLY include the keywords of the assocation. 
    #                     Otherwise, respond with No and No only.
    #                     '''
    #             response = self.llm.invoke(prompt).content

    #             print(response)
    #             if response.lower() != 'no':
    #                 associations.append((f, s, response))

    #     return associations


    def identify_dependencies(self, build_using: str = 'concepts', dependencies_from: str = 'chapters') -> dict[str, list[str]]:
        '''
        Identify the dependency relationships between chapters or topics 

        Args: 
            build_using (str, default concepts): what build dependencies with. options: concepts, outcomes
            dependencies_from (str, default chapters): what to be dependencies on. (chapters/topics)

        Returns:
            dict[str, list[str]]: depedencies between chapters as adjacency list
        '''
        if build_using != 'concepts' and build_using != 'outcomes':
            raise ValueError('build_using must be concepts or outcomes')

        relation = ''

        
        if dependencies_from.lower() == 'chapters':
            relations_dict = create_concept_graph_structure(self.chapters)
            for i in range(len(self.chapters)):
                # current_concept = ' '.join(content[i])
                current = ' '.join(self.concepts[i] if build_using == 'concepts' else self.outcomes[i])
                for j in range(i + 1, len(self.chapters)):
                    # next_concept = ' '.join(content[j])
                    next_ = ' '.join(self.concepts[j] if build_using == 'concepts' else self.outcomes[j])

                    relation = self.llm.generate(f"Identify if these {build_using}: {next_} are prerequisites for these {build_using}: {current}. If there is NO prerequisite, respond with 'No' and 'No' only.")
                    if relation.lower() != 'no':
                        relations_dict[self.chapters[j]].append(self.chapters[i])
        else:
            relations_dict = create_concept_graph_structure(self.main_topics)
            for i in range(len(self.main_topics)):
                current = ' '.join(self.topic_key_points[i] if build_using == 'concepts' else self.topic_objectives[i])
                for j in range(i + 1, len(self.main_topics)):
                    next_ = ' '.join(self.topic_key_points[j] if build_using == 'concepts' else self.topic_objectives[j])

                    relation = self.llm.generate(f"Identify if these {build_using}: {next_} are prerequisites for these {build_using}: {current}. If there is NO prerequisite, respond with 'No' and 'No' only.")
                    if relation.lower() != 'no':
                        relations_dict[self.main_topics[j]].append(self.main_topics[i])

        self.dependencies = relations_dict
        return relations_dict


    @staticmethod
    def draw_graph(data: dict[str, list[str]]) -> None:
        '''
        Print a directed graph using the dependency dictionary

        Args:
            concept_graph: the dictionary to build the graph from
        
        Returns:
            None
        '''
        graph = graphviz.Digraph()

        for key, values in data.items():
            graph.node(name = key)
            for value in values:
                graph.edge(key, value)

        display(Image(graph.pipe(format = "png", renderer = "cairo")))


    # NOTE: I think I might be able to combine these two functions into one
    # def get_assocation_interactive_graph(self, graph: dict[str, tuple[str, str]], associations: dict[str, list[str]]) -> Network:
    #     '''
    #     Retrieve the interactive graph using the association dictionary. Nodes are chapter names and edges are the associations. Hovering over a node results in displaying that nodes learning outcomes and concepts. The function is not able to automatically display the graph so the .show() method must be called on the return object

    #     Args:
    #         graph: The dictionary containing chapter names as keys and the values as a tuple containing the concept at index 0 and outcome at index 1. Can be created automatically using the create_chapter_dict function
    #         assocations: The dictionary containing the associations between chapters. Can be created automatically using the identify_associations() function
        
    #     Returns:
    #         A pyvis Network object
    #     '''

    #     graph = Network(notebook = True, cdn_resources = "remote")

    #     graph.toggle_physics(False)

    #     # Showing all interactivity options, but can be parameterized to only include some
    #     graph.show_buttons()
        
    #     node_id_dict = get_node_id_dict(graph)

    #     for chapter_name, chapter_id in node_id_dict.items():
    #         graph.add_node(n_id = chapter_id, label = chapter_name, title = "Main Learning Concepts: " + graph[chapter_name][0] + "\n" + "Main Learning Outcomes:" + graph[chapter_name][1])

    #     for key, values in associations.items():
    #         for value in values:
    #             graph.add_edge(node_id_dict[key], node_id_dict[value])

    #     return graph


    # def get_dependency_interactive_graph(self, dependency_dict: dict[str, list[str]]) -> Network:
        # '''
        # Retrieve the interactive graph using the dependency dictionary. The function is not able to automatically display the graph so the .show() method must be called on the return object

        # Args:
        #     dependency_dict (dict[str, list[str]]): A dictionary containing the chapter_names between chapters. Can be created automatically using the dependency_relation_extraction() function
        
        # Returns:
        #     A pyvis Network object
        # '''

        # dependency_graph = Network(notebook = True, cdn_resources = "remote")
        # dependency_graph.toggle_physics(False)
        # dependency_graph.show_buttons()

        # node_id_dict = get_node_id_dict(dependency_dict)

        # for chapter_name, chapter_id in node_id_dict.items():
        #     dependency_graph.add_node(n_id = chapter_id, label = chapter_name)

        # for key, values in dependency_dict.items():
        #     for value in values:
        #         dependency_graph.add_edge(node_id_dict[key], node_id_dict[value])

        # return dependency_graph
    
    @staticmethod
    def draw_hypergraph(data: dict[str, list[str]]) -> None:
        '''
        Generate and display a hypergraph

        Args:
            data: (dict[str, list[str]]): data to build hypergraph from

        Returns:
            None
        '''     
        sorted_dependencies = graphlib.TopologicalSorter(data)
        sorted_dependencies = tuple(sorted_dependencies.static_order())

        temp = sorted_dependencies
        sorted_dependencies = {}

        for value in temp:
            sorted_dependencies[value] = data[value]

        hypergraph = hnx.Hypergraph(sorted_dependencies)
        hnx.draw(hypergraph)
        plt.title("Hypergraph")
        plt.show()


    # def draw_layered_graph(self, dictionary: dict[str, list[str]]) -> None:
    #     '''
    #     Generate and display a multilayered graph given a dependency dictionary generated from identify_dependencies() or association dict generated by identify_associations()

    #     Args:
    #         dependencies (dict[str, list[str]]): A dictionary of dependencies. Generated by identify_dependencies(). The key should be a chapter name and the value a list of chapters it depends on

    #     Returns:
    #         None
    #     '''

    #     sorted_dependencies = graphlib.TopologicalSorter(graph = dictionary)
    #     sorted_dependencies = tuple(sorted_dependencies.static_order())

    #     temp = sorted_dependencies
    #     sorted_dependencies = {}

    #     for value in temp:
    #         sorted_dependencies[value] = dictionary[value]


    # def draw_hierarchy(self) -> None:
    #     '''
    #     Draws the hierarchy given a list of terms

    #     Args:
    #         terms (list[str]): terms to draw a hierarchy of

    #     Returns:
    #         None
    #     '''
    #     if self.terminology is None:
    #         raise AttributeError('self.terminiology not found, run build_terminology first')

    #     # network = Network(notebook = True, cdn_resources = 'remote')
    #     tree = graphviz.Digraph()

    #     for first, second in self.terminology:
    #         tree.node(name = first)
    #         tree.node(name = second)
    #         tree.edge(second, first)
        
    #     display(Image(tree.pipe(format = "png", renderer = "cairo", engine = 'dot')))

        # nodes = set()
        # edges = set()
        # node_ids = {}

        # i = 0 # for node ids # for node ids
        # for first, second in self.terminology:
        #     if first not in nodes:
        #         network.add_node(i, label = first)
        #         nodes.add(first)
        #         node_ids[first] = i
        #         i += 1

        #     if second not in nodes:
        #         network.add_node(i, label = second)
        #         nodes.add(second)
        #         node_ids[second] = i
        #         i += 1

        #     if (first, second) not in edges:
        #         edges.add((node_ids[first], node_ids[second]))
        #         network.add_edge(node_ids[first], node_ids[second])

        # if not os.path.exists('./visualizations/'):
        #     os.mkdir('./visualizations/')

        # network.repulsion()
        # display(network.show('./visualizations/tree_hierarchy.html'))


    def evaluate(self, 
                type_eval: str, 
                num_generated: int,
                ground_truth: list[list[str]] | list[str], 
                metrics: list
                ) -> list[dict]:
        ''' 
        Evaluate concepts or outcomes generated from the large language model  

        Args:
            type_eval (str): type of evaluation. 'concepts' to evaluate generated concepts, 'outcomes' to evaluate generated outcomes
            num_generated (int): number of generated concepts or outcomes
            ground_truth (list): ground truth concepts 
            data (list[list[str]] | dict[str, list[str]]): data given to function that identifies terms, concepts, or outcomes
            metrics (list): list of metrics to use for evaluation

        Returns:
            list[dict]: evaluation results per sample
        '''
        if self.concepts is None and self.outcomes is None and self.key_terms is None:
            raise AttributeError('you must have at least one of the following functions ran to perform evaluation: identify_concepts, identify_outcomes, identify_key_terms')

        samples = []
        for i in range(len(self.chapters)):
            if type_eval == 'concepts': # concepts always come from web chapters
                generated = ' '.join(self.concepts[i])
                prompt = f'''
                Given the following context, please identify the {num_generated} most important learning concepts related to {self.chapters[i]}. 
                Your response should directly reference key concepts and terminology from the context provided.

                Context: {self.retrieved_concept_context[list(self.retrieved_concept_context.keys())[i]]}
                
                Additionally, use the following format for your response:
                Concept 1,
                Concept 2,
                Concept 3,
                Concept 4,
                .
                .
                .
                Concept n
                '''           
                retrieved = self.retrieved_concept_context[list(self.retrieved_concept_context.keys())[i]]

            elif type_eval == 'outcomes':
                generated = ' '.join(self.outcomes[i])
                prompt = f'Identify the {num_generated} most important learning outcomes for chapter {self.chapters[i]}. The relevant context can be found here: {self.retrieved_outcome_context[list(self.retrieved_outcome_context.keys())[i]]}.'
                retrieved = self.retrieved_outcome_context[list(self.retrieved_outcome_context.keys())[i]]
            
            elif type_eval == 'terms':
                generated = ' '.join(self.key_terms[i])
                prompt = f'Identify {num_generated} key terms for chapter {self.chapters[i]}. The relevant context can be found here: {self.retrieved_term_context[list(self.retrieved_term_context.keys())[i]]}'
                retrieved = self.retrieved_term_context[list(self.retrieved_term_context.keys())[i]]

            samples.append(LLMTestCase(
                input = prompt, 
                actual_output = generated,
                retrieval_context = retrieved,
                expected_output = ' '.join(ground_truth[i]) if isinstance(ground_truth[0], list) else ' '.join(ground_truth),
            ))

        results = []
        for sample in samples: # i would like to parallelize this but python is annoying about parallelization
            for metric in metrics:
                metric.measure(sample)
                result = {
                    'name': metric.__name__,
                    'score': metric.score,
                    'input': sample.input,
                    'output': sample.actual_output,
                    'success': metric.is_successful(),
                    'reason': metric.reason,
                    'expected': sample.expected_output,
                }
                results.append(result)

        return results


    # def build_terminology(self, build_using: str = 'concepts') -> list[tuple[str, str]]:
    #     '''
    #     Build terminology using key terms and is-a relationships 

    #     Args:
    #         build_using (str, default concepts):

    #     Returns:
    #         list[tuple[str, str]]: is-a relationships 
    #     '''
    #     if not self.key_terms and not self.concepts:
    #         # should i do it for them here
    #         raise AttributeError('identify_key_terms or identify_concepts must be ran first')

    #     terminology = set()
    #     data = [clean(word) for l in (self.concepts if build_using == 'concepts' else self.key_terms) for word in l]

    #     with ThreadPoolExecutor(max_workers = min(4, len(data))) as p:
    #         futures = []
    #         for i in range(len(data)):
    #             for j in range(len(data)):
    #                 if i != j:
    #                     future = p.submit(process_pair, data[i], data[j], self.llm)
    #                     futures.append(future)

    #         for f in futures:
    #             if f.result() is not None:
    #                 terminology.add(f.result())

    #     self.terminology = list(terminology)
    #     return list(terminology)


    @staticmethod
    def build_knowledge_graph(dependencies: dict[str, list[str]], concepts: list[list[str]], outcomes: list[list[str]]) -> Network:
        '''
        Builds a knowledge using learning concepts, outcomes, and key terms

        Args:
            dependencies (dict[str, list[str]]): topic or chapter dependencies 
            concepts (list[list[str]]): some form of concepts/terms associated with each chapter or topic
            outcomes (list[list[str]]): some form of outcomes/objectives associated with chapter or topic 
        
        Returns:
            Network: pyvis network
        '''
        kg = Network(notebook = True, cdn_resources='remote')
        
        keys = list(dependencies.keys())
        used_nodes = {}

        j = 0 # used for node ids
        for k, v in dependencies.items():
            if k not in used_nodes:
                kg.add_node(j, k, color = 'red', size = 4) # node id, node label
                used_nodes[k] = j
                j += 1
                
            for neighbor in v:
                if neighbor not in used_nodes:
                    kg.add_node(j, neighbor, 'red')
                    used_nodes[neighbor] = j
                    j += 1
                
                kg.add_edge(used_nodes[k], used_nodes[neighbor], label = 'requires')

        for i in range(len(concepts)):
            concept_str = ' '.join(concepts[i])
            kg.add_node(j, f'{keys[i]} concepts', title = concept_str, color = 'green', size = 2)
            kg.add_edge(j, used_nodes[keys[i]], label = 'covers')
            j += 1
    
            # terms = ' '.join(self.key_terms[i])
            # kg.add_node(j, f'{keys[i]} terms', title = terms, color = 'blue', size = 2)
            # kg.add_edge(j, used_nodes[keys[i]], label = 'contains key terms')
            # j += 1

            outcome = ' '.join(outcomes[i])
            kg.add_node(j, f'{keys[i]} outcomes', title = outcome, color = 'purple', size = 2)
            kg.add_edge(j , used_nodes[keys[i]], label = 'results with knowledge in')
            j += 1
    
        return kg


    @staticmethod
    def draw_knowledge_graph(kg: Network, save_to: str) -> None:
        '''
        Turns knowledge graph into an html file that can be visualized, file path: ./visualizations/kg.html

        Args:
            kg (Network): pyvis network to visualize to
            save_to (str): path to save html file to

        Returns:
            None
        '''
        kg.repulsion(spring_length = 250)
        display(kg.show(save_to))


    # def visualize_results(self, results_dict: list[dict]) -> None:
    #     '''
    #     Visualize results after evaluation using heatmap

    #     Args:
    #         results_dict (list[dict]): list of dictionaries with test case results

    #     Returns:
    #         None
    #     '''
    #     p_scores = [d['score'] for d in results_dict if d['name'] == 'Contextual Precision']
    #     recall_scores = [d['score'] for d in results_dict if d['name'] == 'Contextual Recall']
    #     ar_scores = [d['score'] for d in results_dict if d['name'] == 'Answer Relevancy']
    #     ac_scores = [d['score'] for d in results_dict if d['name'] == 'Answer Correctness']
    #     s_scores = [d['score'] for d in results_dict if d['name'] == 'SemanticSimilarity']
    #     f_scores = [d['score'] for d in results_dict if d['name'] == 'Faithfulness']

    #     figs, ax = plt.subplots(1, 1)
    #     ax[0].plot(p_scores, label = 'Contextual Precison')
    #     ax[0].plot(recall_scores, label = 'Contextual Recall')
    #     ax[0].plot(ar_scores, label = 'Answer Relevancy')
    #     ax[0].plot(ac_scores, label = 'Answer Correctness')
    #     ax[0].plot(s_scores, label = 'Semantic Similarity')
    #     ax[0].plot(f_scores, label = 'Faithfulness')
    #     ax[0].set_title('metric scores across chapters')
    #     ax[0].set_xlabel('Chapter')
    #     ax[0].set_ylabel('Score')
        
    #     plt.legend()
    #     plt.show()

    #     if not os.path.exists('./visualizations/'):
    #         os.mkdir('./visualizations/')

    #     plt.savefig('./visualizations/results.png')
        