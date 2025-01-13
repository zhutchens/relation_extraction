from langchain_openai.chat_models import ChatOpenAI
import graphviz
import os
import re
from IPython.display import display, Image
from pyvis.network import Network
import hypernetx as hnx
import matplotlib.pyplot as plt
import graphlib
# from ragas import evaluate as ev
# from ragas import SingleTurnSample
# from ragas.dataset_schema import EvaluationDataset
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from src.retrieval import RetrievalSystem
from langchain_core.documents import Document
from src.utils import rank_docs
from string import punctuation  
from concurrent.futures import ThreadPoolExecutor
from deepeval import evaluate as ev
# from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import EvaluationResult
from ipysigma import Sigma


def create_concept_graph_structure(param_list: list) -> dict:
    return_dict = {}

    for list_itm in param_list:
        return_dict[list_itm] = []

    return return_dict


def get_node_id_dict(dictionary: dict) -> dict:
    
    node_id_dict = {}
    id_count = 1
    
    for name in dictionary.keys():
        node_id_dict[name] = id_count
        id_count += 1

    return node_id_dict    


# NOTE: Currently one main issue, the function that finds the associations between chapters seems to be broken. I think its the algorithm thats wrong. It also takes 10+ minutes to run
class relationExtractor:
    '''
    
    '''
    def __init__(self, 
                link: str, 
                token: str, 
                chapters: list[str], 
                connection_string: str,
                chunk_size: int,
                chunk_overlap: int,
                db_name: str,
                collection_name: str,
                reset: bool = False,
                openai_model: str = 'gpt-4o-mini',
                st_model: str = 'all-MiniLM-L12-v2',
                temp: int = 0):
        '''
        Constructor to create a large language model relation extractor class. 

        Args:
            link (str): the web link to generate answers from
            token (str): OpenAI token
            chapters (list): list of chapter/section names in link
            connection_string (str): connection string to MongoDB 
            chunk_size (int): number of characters in a chunk of content
            chunk_overlap (int): number characters to overlap between content chunks
            db_name (str): name of MongoDB database
            collection_name (str): collection name of MongoDB database
            reset (bool, default False): if True, drop all documents from collection before adding new ones
            model (str, default gpt-4o-mini): OpenAI model to use 
            temp (int, default 1): temperature to use with OpenAI model
        '''
        self.link = link
        self.chapters = chapters
        self.openai_model = openai_model
        self.embedding_model = st_model

        os.environ['OPENAI_API_KEY'] = token
        self.llm = ChatOpenAI(model = openai_model, max_tokens = None, temperature = temp)

        self.vs = RetrievalSystem(connection_string, self.link, chunk_size, chunk_overlap, db_name, collection_name, st_model, reset)

    
    def retrieval_pipeline(self, query: str, context: str, num_queries: int = 4, top_n: int = 4) -> list[Document]:
        '''
        Retrieval pipeline from original query

        Args:
            query (str): original prompt
            context (str): (something)
            num_queries (int, default 4): number of additional queries to generate
            top_n (int, default 4): top n documents to return

        Returns:
            list[Document]: top_n documents
        '''
        q_prompt = f'''
                    You are tasked with enhancing this {query} for a retrieval-augmented pipeline. Output {num_queries} additional queries.
                    You can find relevant context here: {context}.

                    Output Format:
                    query_1::query_2::query_3::query_4...::query_{num_queries}
                    '''
        
        queries = self.llm.invoke(q_prompt).content.split('::')

        retrieved_docs = []
        for q in queries:
            q_docs = self.vs.invoke(q)

            for doc in q_docs:
                retrieved_docs.append((q, doc))

        # rank and return top_n docs
        return rank_docs(retrieved_docs, top_n = top_n)
        

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
        #             Identify {n_terms} key terms from Chapter {chapter_name} in descending order of significance, listing the most significant terms first. The textbook is available here: {self.link}.
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
        terms = []
        retrieved = {}
        for chapter in self.chapters:
            relevant_docs = self.retrieval_pipeline(f'Identify {n_terms} key terms for chapter {chapter}.', self.link)
            retrieved[chapter] = relevant_docs

            prompt = f'''
                    Identify {n_terms} key terms for chapter {chapter}. 
                    Relevant documents can be found here: {relevant_docs}

                    Provide your answer in the following format:

                    Concept 1,
                    Concept 2,
                    Concept 3, 
                    ...
                    Concept n
                    '''

            words = self.llm.invoke(prompt).content
            terms.append([string for string in words.split('\n') if string != ''])

        return terms, retrieved


    def summarize(self) -> str:
        '''
        Returns summary of object web link

        Args:
            None

        Returns:
            str: summary of textbook
        '''
        return self.llm.invoke(f"Please summarize this web page: {self.link}").content


    def create_chapter_dict(self, outcomes: list[str], concepts: list[str]) -> dict[str, tuple[str, str]]:
        '''
        Create a chapter dictionary containing the chapter names as keys and its concepts and outcomes in a tuple as the value

        Args:
            outcomes (list[str]): a list of learning outcomes created from the identify_learning_outcomes function
            concepts (list[str]): a list of learning concepts created from the identify_learning_concepts function

        Returns:
            dict[str, tuple[str, str]]: dictionary containing chapter names as keys and concepts/outcomes as tuple
        '''
        outcome_concept_graph = {}

        for idx, name in enumerate(self.chapters):
            outcome_concept_graph[name] = (concepts[idx], outcomes[idx])
                
        return outcome_concept_graph
        
    # note: fix later
    # def identify_chapters(self) -> dict[str, str]:
    #     '''
    #     Identify the chapters within the class provided link using a large language model
    #     It is very, very inconsistent and I highly recommened manually creating the dictionary

    #     Args:
    #         None

    #     Returns:
    #         dict[str, str]: key is chapter number, value is chapter name
    #     '''
    #     # NOTE: This is very, very, inconsistent. Do not recommend using this.

    #     # chapters = self.llm.invoke(f"Please identify the chapters in this textbook: {self.link}").content
    #     chapters = self.llm.invoke().content
    #     chapters = chapters.split('\n')
    #     chapter_dict = {}
        
    #     for idx, chapter in enumerate(chapters):
    #         chapter_dict[f"Chapter {idx + 1}"] = chapter

    #     return chapter_dict


    def identify_main_topics(self) -> list[str]:
        '''
        Identify the main topics within the class provided link

        Args:
            None
        
        Returns:
            list[str]: list of main topics 
        '''

        prompt = f'''
                    Please identify the main topics from this textbook: {self.link}. Please provide justification.
                    Format:
                    main topic :: justification
                  '''
        main_topics = self.llm.invoke(prompt).content

        return [topic for topic in main_topics.split('\n') if topic != '']
    
    
    def identify_main_topic_relations(self, main_topic_list: list[str]) -> dict[str, list[str]]:
        '''
        Identify the relationships between the main topics of the textbook

        Args:
            main_topics_list (list[str]): A list of main topics from the textbook. Can be automatically created using the identify_main_topics() function
        
        Returns:
            dict[str, list[str]]: relationships between main topics as adjacency list
        '''
        topic_relations = create_concept_graph_structure(main_topic_list)

        relation = ''
        for i in range(len(main_topic_list)):
            for j in range(len(main_topic_list)):
                if i != j:
                    relation = self.llm.invoke(f"Is there a relationship between this topic: {main_topic_list[i]}, and this topic: {main_topic_list[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split(',')[0].strip() != 'No':
                        topic_relations[main_topic_list[i]].append(main_topic_list[j])

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
        outcome_list = []
        retrieved_contexts = {}
        # if concepts is None and concepts_keyTerms is None:

        for name in self.chapters:
            prompt = f'Identify {num_outcomes} learning outcomes from chapter {name}.'
            relevant_docs = self.retrieval_pipeline(prompt, self.link)
        
            # # single shot prompt 
            single_prompt = f'''
                            Identify the {num_outcomes} most important learning outcomes for chapter: {name}. 
                            The relevant context can be found here: {' '.join(t[0][1].page_content for t in relevant_docs)}
                            '''

            retrieved_contexts[name] = ''.join([t[0][1].page_content for t in relevant_docs])

            response = self.llm.invoke(single_prompt).content

            outcome_list.append([outcome for outcome in response.split('\n') if outcome != ''])

        return outcome_list, retrieved_contexts
        
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
        concept_list = []
        current_concept = ''
        retrieved_contexts = {}

        for name in self.chapters:
            prompt = f'Identify {num_concepts} learning concepts from chapter {name}.'
            relevant_docs = self.retrieval_pipeline(prompt, self.link)
            
            # # single shot prompt 
            single_prompt = f'''
                             Identify the {num_concepts} most important learning concepts for chapter: {name}. 
                             The relevant context can be found here: {' '.join(t[0][1].page_content for t in relevant_docs)}
                             '''

            retrieved_contexts[name] = ''.join([t[0][1].page_content for t in relevant_docs])

            current_concept = self.llm.invoke(single_prompt).content
            concept_list.append([concept for concept in current_concept.split('\n') if concept != ''])

        return concept_list, retrieved_contexts

    
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


    def identify_dependencies(self, content: list[list[str]]) -> dict[str, list[str]]:
        '''
        Identify the dependency relationships between chapters

        Args: 
            content (list[list[str]]): key terms, outcomes, or concepts to be used to identify chapter dependencies 

        Returns:
            dict[str, list[str]]: depedencies between chapters as adjacency list
        '''

        relation = ''

        relations_dict = create_concept_graph_structure(self.chapters)

        for i in range(len(self.chapters)):
            current_concept = ' '.join(content[i])
            for j in range(i + 1, len(self.chapters)):
                next_concept = ' '.join(content[j])

                relation = self.llm.invoke(f"Please identify if these concepts: {next_concept} are a prerequisite for these concepts: {current_concept}. If there is NO prerequisite, please respond with 'No' and 'No' only.").content
                if relation.split(',')[0].strip() != 'No':
                    relations_dict[self.chapters[j]].append(self.chapters[i])

        return relations_dict


    def draw_graph(self, concept_graph: dict[str, list[str]]) -> None:
        '''
        Print a directed graph using the dependency dictionary

        Args:
            concept_graph: The dictionary to build the graph from. This should come from either the identify_associations function or identify_dependencies function
        
        Returns:
            None
        '''
        graph = graphviz.Digraph()

        for key, values in concept_graph.items():
            graph.node(name = key)
            for value in values:
                graph.edge(key, value)

        display(Image(graph.pipe(format = "png", renderer = "cairo")))


    # NOTE: I think I might be able to combine these two functions into one
    def get_assocation_interactive_graph(self, graph: dict[str, tuple[str, str]], associations: dict[str, list[str]]) -> Network:
        '''
        Retrieve the interactive graph using the association dictionary. Nodes are chapter names and edges are the associations. Hovering over a node results in displaying that nodes learning outcomes and concepts. The function is not able to automatically display the graph so the .show() method must be called on the return object

        Args:
            graph: The dictionary containing chapter names as keys and the values as a tuple containing the concept at index 0 and outcome at index 1. Can be created automatically using the create_chapter_dict function
            assocations: The dictionary containing the associations between chapters. Can be created automatically using the identify_associations() function
        
        Returns:
            A pyvis Network object
        '''

        graph = Network(notebook = True, cdn_resources = "remote")

        graph.toggle_physics(False)

        # Showing all interactivity options, but can be parameterized to only include some
        graph.show_buttons()
        
        node_id_dict = get_node_id_dict(graph)

        for chapter_name, chapter_id in node_id_dict.items():
            graph.add_node(n_id = chapter_id, label = chapter_name, title = "Main Learning Concepts: " + graph[chapter_name][0] + "\n" + "Main Learning Outcomes:" + graph[chapter_name][1])

        for key, values in associations.items():
            for value in values:
                graph.add_edge(node_id_dict[key], node_id_dict[value])

        return graph


    def get_dependency_interactive_graph(self, dependency_dict: dict[str, list[str]]) -> Network:
        '''
        Retrieve the interactive graph using the dependency dictionary. The function is not able to automatically display the graph so the .show() method must be called on the return object

        Args:
            dependency_dict (dict[str, list[str]]): A dictionary containing the chapter_names between chapters. Can be created automatically using the dependency_relation_extraction() function
        
        Returns:
            A pyvis Network object
        '''

        dependency_graph = Network(notebook = True, cdn_resources = "remote")
        dependency_graph.toggle_physics(False)
        dependency_graph.show_buttons()

        node_id_dict = get_node_id_dict(dependency_dict)

        for chapter_name, chapter_id in node_id_dict.items():
            dependency_graph.add_node(n_id = chapter_id, label = chapter_name)

        for key, values in dependency_dict.items():
            for value in values:
                dependency_graph.add_edge(node_id_dict[key], node_id_dict[value])

        return dependency_graph
    

    def draw_hypergraph(self, dictionary: dict[str, list[str]]) -> None:
        '''
        Generate and display a hypergraph given a dependency dictionary generated from identify_dependencies() or association dict generated by identify_associations()

        Args:
            dependencies (dict[str, list[str]]): A dictionary of dependencies. Can be generated by identify_dependencies(). The key should be a chapter name and the value a list of chapters it depends on

        Returns:
            None
        '''
                
        sorted_dependencies = graphlib.TopologicalSorter(graph = dictionary)
        sorted_dependencies = tuple(sorted_dependencies.static_order())

        temp = sorted_dependencies
        sorted_dependencies = {}

        for value in temp:
            sorted_dependencies[value] = dictionary[value]

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


    def draw_hierarchy(self, terms: list[tuple[str, str]]) -> None:
        '''
        Draws the hierarchy given a list of terms

        Args:
            terms (list[str]): terms to draw a hierarchy of

        Returns:
            None
        '''
        g = graphviz.Digraph(graph_attr = {'rankdir': 'TB'})
        nodes = set()
        edges = set()

        for tup in terms:
            first, second = tup
            if first not in nodes:
                g.node(first)

                if (first, second) not in edges:
                    g.edge(first, second)

                nodes.add(first)
                edges.add((first, second))

        display(Image(g.pipe(format = "png", renderer = "cairo", engine = 'dot')))


    def evaluate(self, 
                type_eval: str, 
                num_generated: int,
                generated: list[list[str]] | dict[str, list[str]], 
                ground_truth: list[str], 
                data: list[list[str]] | dict[str, list[str]],
                metrics: list, 
                print_results: bool = False
                ) -> EvaluationResult:
                # ) -> list[SingleTurnSample]:
        ''' 
        Evaluate concepts or outcomes generated from the large language model  

        Args:
            type_eval (str): type of evaluation. 'concepts' to evaluate generated concepts, 'outcomes' to evaluate generated outcomes
            num_generated (int): number of generated concepts or outcomes
            generated (list[list[str]] | dict[str, list[str]]): dictionary with chapter name as key and value as the list of concepts
            ground_truth (list): ground truth concepts 
            data (list[list[str]] | dict[str, list[str]]): data given to function that identifies terms, concepts, or outcomes
            metrics (list): list of metrics to use for evaluation
            print_results (bool, default False): if True, prints results to screen

        Returns:
            EvaluationResult: evaluation results 
        '''      
        samples = []
        if isinstance(data, dict):
            keys = list(data.keys())

        if isinstance(generated, dict):
            values = list(generated.values())

        for i in range(len(ground_truth)):
            if type_eval == 'concepts': # concepts always come from web chapters
                prompt = f'Identify the {num_generated} most important learning concepts for chapter {self.chapters[i]}. The relevant context can be found here: {data[keys[i]]}.'            
                retrieved = data[keys[i]]

            elif type_eval == 'outcomes':
                prompt = f'Identify the {num_generated} most important learning outcomes for chapter {self.chapters[i]}. The relevant context can be found here: {data[keys[i]]}.'
                retrieved = data[keys[i]]
            
            elif type_eval == 'terms':
                concept = ' '.join(data[i])
                prompt = f'Identify {num_generated} key terms for the following concept: {concept}.'
                retrieved = data[keys[i]]


            samples.append(LLMTestCase(
                input = prompt, 
                actual_output = ' '.join(generated[i]) if not isinstance(generated, dict) else ' '.join(values[i]),
                retrieval_context = [retrieved],
                expected_output = ground_truth[i],
                # reference_contexts = [textbook[self.chapters[i]]] # for now this is just the chapter text, maybe should remove
            ))

        # dataset = EvaluationDataset(test_cases = samples)
        result = ev(samples, metrics, print_results = print_results)
        return result


    def build_terminology(self, terms: list[list[str]], num_workers: int = 1) -> list[tuple[str, str]]:
        '''
        Build terminology using terms and is-a relationships 

        Args:
            terms (list[list[str]]): key terms to use
            num_workers (int, default 1): number of workers to use 

        Returns:
            list[tuple[str, str]]: is-a relationships 
        '''
        def clean(text):
            text = ''.join([char for char in text if char not in punctuation])
            text = text.strip()
            return text

        def process_pair(first, second):
            if first == second: return None
            prompt = f'''
                Q: Is there an is-a relationship present between dog and mammal?
                A: Yes

                Q: Is there an is-a relationship present between vehicle and car?
                A: No

                Q: Is there an is-a relationship presnet between {first} and {second}?
                '''
            response = self.llm.invoke(prompt).content
            return (first, second) if 'yes' in response.lower() else None

        terminology = set()
        terms = [clean(word) for l in terms for word in l]

        with ThreadPoolExecutor(max_workers = num_workers) as p:
            futures = []
            for i in range(len(terms)):
                for j in range(len(terms)):
                    if i != j:
                        futures.append(p.submit(process_pair, terms[i], terms[j]))

            for f in futures:
                result = f.result()
                if result is not None: terminology.add(result)

        return list(terminology)


    def build_knowledge_graph(self, dependencies: dict[str, list[str]], concepts: list[list[str]], outcomes: list[list[str]], key_terms: list[list[str]]) -> nx.Graph:
        '''
        Builds a knowledge using learning concepts, outcomes, and key terms

        Args:
            dependencies (dict[str, list[str]]): chapter dependencies 
            concepts (list[list[str]]): learning concepts
            outcomes (list[list[str]]): learning outcomes
            key_terms (list[list[str]]): key_terms for chapters
        
        Returns:
            nx.Graph: networkx object 
        '''
        kg = Network(notebook = True, cdn_resources='remote')
        
        keys = list(dependencies.keys())
        used_nodes = {}

        j = 0 # used for node ids
        for k, v in dependencies.items():
            if k not in used_nodes:
                kg.add_node(j, k, color = 'red') # node id, node label
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
            kg.add_node(j, f'{keys[i]} concepts', title = concept_str, color = 'green')
            kg.add_edge(j, used_nodes[keys[i]], label = 'covers')
            j += 1
    
            terms = ' '.join(key_terms[i])
            kg.add_node(j, f'{keys[i]} terms', title = terms, color = 'blue')
            kg.add_edge(j, used_nodes[keys[i]], label = 'contains key terms')
            j += 1

            outcome = ' '.join(outcomes[i])
            kg.add_node(j, f'{keys[i]} outcomes', title = outcome, color = 'purple')
            kg.add_edge(j , used_nodes[keys[i]], label = 'results with knowledge in')
            j += 1
    
        return kg

    
    def draw_knowledge_graph(self, kg: nx.Graph, return_html: bool = False) -> None:
        '''
        Takes knowledge graph and turns it into an html file that can be used to visualize a knowledge graph

        Args:
            kg (nx.Graph): networkx graph to draw
            return_html (bool, default False): if True, does not display the graph and returns it as an html file. Saves it into a visualizations folder as kg.html

        Returns:
            None
        '''
        # if not return_html:
        #     widget = Sigma(kg, node_color = 'color', default_edge_type = 'curve', node_border_color_from = 'node')
        #     display(widget)
        #     return

        if not os.path.exists('./visualizations/'):
            os.mkdir('./visualizations/')

        # Sigma.write_html(Sigma(kg, node_color = "tag", node_label_size = kg.degree, node_size = kg.degree), './visualizations/kg.html')
        # kg.show_buttons(filter_=["physics"])
        kg.repulsion(spring_length = 250)
        display(kg.show('./visualizations/kg.html'))


    # def generate_testset(self, size: int = 10) -> DataFrame:
    #     '''
    #     Generates a testset for evaluation based on given link

    #     Args:
    #         size (int, default 10): number of samples

    #     Returns:
    #         DataFrame: dataframe of samples
    #     '''
    #     gen = TestsetGenerator(LangchainLLMWrapper(self.llm), 
    #                           LangchainEmbeddingsWrapper(TransformerEmbeddings()))

    #     ds = gen.generate_with_langchain_docs(documents = self.vs.docs, testset_size = size)

    #     return ds.to_pandas()
    