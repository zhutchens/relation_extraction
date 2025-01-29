from langchain_openai.chat_models import ChatOpenAI
import graphviz
import os
import re
from IPython.display import display, Image
from pyvis.network import Network
import hypernetx as hnx
import matplotlib.pyplot as plt
import graphlib
from src.retrieval import RetrievalSystem
from deepeval.test_case import LLMTestCase
from src.utils import create_concept_graph_structure, clean, process_pair
from concurrent.futures import ThreadPoolExecutor
from src.utils import normalize_text
from deepeval.models import DeepEvalBaseLLM


# NOTE: Currently one main issue, the function that finds the associations between chapters seems to be broken. I think its the algorithm thats wrong. It also takes 10+ minutes to run
class relationExtractor:
    '''
    
    '''
    def __init__(self, 
                documents: list[str] | str,  
                chapters: list[str], 
                chunk_size: int,
                chunk_overlap: int,
                llm: DeepEvalBaseLLM, 
                st_model: str = 'msmarco-distilbert-base-tas-b',
                temp: int = 0):
        '''
        Constructor to create a large language model relation extractor class. 

        Args:
            documents (list[str] or str): the string, web link, or pdf path to get information from
            chapters (list): list of chapter/section names in link
            chunk_size (int): number of characters in a chunk of content
            chunk_overlap (int): number characters to overlap between content chunks
            llm (DeepEvalBaseLLM): LLM to use. Use one the preconfigured language model classes from src.llms and pass in your model name (optional)
            st_model (str, default msmarco-distilbert-base-tas-b'): sentence transformer to use 
            temp (int, default 1): temperature to use with OpenAI model
        '''
        if not isinstance(documents, list) and not isinstance(documents, str): 
            raise ValueError(f'document must be of type list or str, got type {type(documents)}')

        self.documents = documents
        self.chapters = chapters
        self.temp = temp
        self.embedding_model = st_model
        self.concepts, self.outcomes, self.key_terms, self.retrieved_concept_context, self.retrieved_outcome_context, self.retrieved_term_context, self.terminology, self.kg, self.main_topics, self.dependencies, self.main_topic_relationships, self.summarization = None, None, None, None, None, None, None, None, None, None, None, None

        self.llm = llm 

        self.retriever = RetrievalSystem(self.documents, chunk_size, chunk_overlap, st_model)
        

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

        terms = []
        retrieved = {}
        for chapter in self.chapters:
            relevant_docs = [doc for doc in self.retriever.pipeline(chapter, self.llm)]
            retrieved[chapter] = relevant_docs

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
            terms.append([string for string in words.split('\n') if string != ''])

        self.key_terms = terms
        self.retrieved_term_context = retrieved
        return terms, retrieved


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


    # def create_chapter_dict(self, outcomes: list[str], concepts: list[str]) -> dict[str, tuple[str, str]]:
    #     '''
    #     Create a chapter dictionary containing the chapter names as keys and its concepts and outcomes in a tuple as the value

    #     Args:
    #         outcomes (list[str]): a list of learning outcomes created from the identify_learning_outcomes function
    #         concepts (list[str]): a list of learning concepts created from the identify_learning_concepts function

    #     Returns:
    #         dict[str, tuple[str, str]]: dictionary containing chapter names as keys and concepts/outcomes as tuple
    #     '''
    #     outcome_concept_graph = {}

    #     for idx, name in enumerate(self.chapters):
    #         outcome_concept_graph[name] = (concepts[idx], outcomes[idx])
                
    #     return outcome_concept_graph
        
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

    #     # chapters = self.llm.invoke(f"Please identify the chapters in this textbook: {self.document}").content
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
                    Please identify the main topics from this textbook: {self.document}. Please provide justification.
                    Format:
                    main topic :: justification
                  '''
        main_topics = self.llm.generate(prompt)

        self.main_topics = [topic for topic in main_topics.split('\n') if topic != '']
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
                    relation = self.llm.generate(f"Is there a relationship between this topic: {main_topic_list[i]}, and this topic: {main_topic_list[j]}? If there is NOT, please respond with 'No' and 'No' only.")
                    relation = re.sub(re.compile('^[a-zA-Z\s\.,!?]'), '', relation)
                    if relation.split(',')[0].strip() != 'No':
                        topic_relations[main_topic_list[i]].append(main_topic_list[j])

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
        outcome_list = []
        retrieved_contexts = {}
        # if concepts is None and concepts_keyTerms is None:

        for name in self.chapters:
            prompt = f'Identify {num_outcomes} learning outcomes from chapter {name}.'
            relevant_docs = [doc for doc in self.retriever.pipeline(name, self.llm)]
        
            # # single shot prompt 
            single_prompt = f'''
                            Identify the {num_outcomes} most important learning outcomes for chapter: {name}. 
                            The relevant context can be found here: {relevant_docs}

                            Additionally, use the following format for your response:
                            Outcome 1,
                            Outcome 2,
                            Outcome 3,
                            Outcome 4,
                            .
                            .
                            .
                            Outcome n
                            '''

            retrieved_contexts[name] = relevant_docs

            response = self.llm.generate(single_prompt)

            outcome_list.append([outcome for outcome in response.split('\n') if outcome != ''])

        self.outcomes = outcome_list
        self.retrieved_outcome_context = retrieved_contexts

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


    def identify_concepts(self, num_concepts: int, top_n: int = 4) -> tuple[dict[str, str], list[list[str]]]:
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
            relevant_docs = [doc for doc in self.retriever.pipeline(name, self.llm, top_n)]
            
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

            retrieved_contexts[name] = relevant_docs

            current_concept = self.llm.generate(single_prompt)
            concept_list.append([concept for concept in current_concept.split('\n') if concept != ''])

        self.concepts = concept_list
        self.retrieved_concept_context = retrieved_contexts

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

                relation = self.llm.generate(f"Identify if these concepts: {next_concept} are prerequisites for these concepts: {current_concept}. If there is NO prerequisite, respond with 'No' and 'No' only.")
                if relation.split(',')[0].strip() != 'No':
                    relations_dict[self.chapters[j]].append(self.chapters[i])

        self.dependencies = relations_dict
        return relations_dict


    def draw_graph(self) -> None:
        '''
        Print a directed graph using the dependency dictionary

        Args:
            concept_graph: The dictionary to build the graph from. This should come from either the identify_associations function or identify_dependencies function
        
        Returns:
            None
        '''
        if self.dependencies is None:
            raise AttributeError('self.dependencies not found, run identify_dependencies first')

        graph = graphviz.Digraph()

        for key, values in self.dependencies.items():
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
    

    def draw_hypergraph(self) -> None:
        '''
        Generate and display a hypergraph

        Args:
            None

        Returns:
            None
        '''
        if self.dependencies is None:
            raise AttributeError('self.dependencies not found, run identify_dependencies first')
                
        sorted_dependencies = graphlib.TopologicalSorter(self.dependencies)
        sorted_dependencies = tuple(sorted_dependencies.static_order())

        temp = sorted_dependencies
        sorted_dependencies = {}

        for value in temp:
            sorted_dependencies[value] = self.dependencies[value]

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


    def draw_hierarchy(self) -> None:
        '''
        Draws the hierarchy given a list of terms

        Args:
            terms (list[str]): terms to draw a hierarchy of

        Returns:
            None
        '''
        if self.terminology is None:
            raise AttributeError('self.terminiology not found, run build_terminology first')

        network = Network(notebook = True, cdn_resources = 'remote', layout = 'hierarchical')
        nodes = set()
        edges = set()
        node_ids = {}

        i = 0 # for node ids # for node ids
        for first, second in self.terminology:
            if first not in nodes:
                network.add_node(i, label = first)
                nodes.add(first)
                node_ids[first] = i
                i += 1

            if second not in nodes:
                network.add_node(i, label = second)
                nodes.add(second)
                node_ids[second] = i
                i += 1

            if (first, second) not in edges:
                edges.add((node_ids[first], node_ids[second]))
                network.add_edge(node_ids[first], node_ids[second])

        if not os.path.exists('./visualizations/'):
            os.mkdir('./visualizations/')

        # network.repulsion()
        display(network.show('./visualizations/tree_hierarchy.html'))


    def evaluate(self, 
                type_eval: str, 
                num_generated: int,
                # generated: list[list[str]] | dict[str, list[str]], 
                ground_truth: list[str], 
                # data: list[list[str]] | dict[str, list[str]],
                metrics: list
                ) -> list[dict]:
                # ) -> list[SingleTurnSample]:
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
                Given the following context, please identify the {num_generated} most important learning concepts related to the chapter on {self.chapters[i]}. 
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
                expected_output = ' '.join(ground_truth),
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


    def build_terminology(self) -> list[tuple[str, str]]:
        '''
        Build terminology using key terms and is-a relationships 

        Args:
            terms (list[list[str]]): key terms to use

        Returns:
            list[tuple[str, str]]: is-a relationships 
        '''
        if self.key_terms is None:
            # should i do it for them here
            raise AttributeError('self.key_terms not found. Run identify_key_terms() first')

        terminology = set()
        terms = [clean(word) for l in self.key_terms for word in l]

        with ThreadPoolExecutor(max_workers = len(terms)) as p:
            futures = []
            for i in range(len(terms)):
                for j in range(len(terms)):
                    if i != j:
                        future = p.submit(process_pair, terms[i], terms[j], self.llm)
                        futures.append(future)

            for f in futures:
                if f.result() is not None:
                    terminology.add(f.result())

        self.terminology = list(terminology)
        return list(terminology)


    def build_knowledge_graph(self) -> Network:
        '''
        Builds a knowledge using learning concepts, outcomes, and key terms

        Args:
            None
        
        Returns:
            Network: pyvis network
        '''
        if self.outcomes is None:
            raise AttributeError('self.outcomes not found, run identify_outcomes first')

        if self.concepts is None:
            raise AttributeError('self.concepts not found, run identify_concepts first')

        if self.key_terms is None:
            raise AttributeError('self.key_terms not found, run identify_key_terms first')

        if self.dependencies is None:
            raise AttributeError('self.dependencies not found, run identify_dependencies first')

        kg = Network(notebook = True, cdn_resources='remote')
        
        keys = list(self.dependencies.keys())
        used_nodes = {}

        j = 0 # used for node ids
        for k, v in self.dependencies.items():
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

        for i in range(len(self.concepts)):
            concept_str = ' '.join(self.concepts[i])
            kg.add_node(j, f'{keys[i]} concepts', title = concept_str, color = 'green', size = 2)
            kg.add_edge(j, used_nodes[keys[i]], label = 'covers')
            j += 1
    
            terms = ' '.join(self.key_terms[i])
            kg.add_node(j, f'{keys[i]} terms', title = terms, color = 'blue', size = 2)
            kg.add_edge(j, used_nodes[keys[i]], label = 'contains key terms')
            j += 1

            outcome = ' '.join(self.outcomes[i])
            kg.add_node(j, f'{keys[i]} outcomes', title = outcome, color = 'purple', size = 2)
            kg.add_edge(j , used_nodes[keys[i]], label = 'results with knowledge in')
            j += 1
    
        self.kg = kg
        return kg

    
    def draw_knowledge_graph(self) -> None:
        '''
        Turns knowledge graph into an html file that can be visualized, file path: ./visualizations/kg.html

        Args:
            None

        Returns:
            None
        '''
        if self.kg is None:
            raise AttributeError(f'self.kg not found. Run build_knowledge_graph() first')

        if not os.path.exists('./visualizations/'):
            os.mkdir('./visualizations/')

        self.kg.repulsion(spring_length = 250)
        display(self.kg.show('./visualizations/kg.html'))


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
    
