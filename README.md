# Self Evaluation of RAG Pipeline
In this repo, I cover 
* The basics of developing a RAG based Question-Answering pipeline using LangChain.
* Evaluate different strategies for chunking text data for creating vector database
* Implement a self-evaluation pipeline to evaluate the QA - RAG pipeline
* Experiment with different chunking/splitting parameters to identify optimal settings


## Main Files


* main.py 
    * Primary script containing key classes/functions
        * **retrieve_context** : Function that takes as input a set of documents and a question, then retrieves the most relevant 'chunks' of text from the docs that help answer the question
        * **Experiment : **Class to help run LLM based self evaluation of the quality of the chunks retrieved
* Main.ipynb
    * Primary notebook for testing retrieve_context and Experiment class
* Experiment Evaluation.ipynb
    * Notebook containing code for running evaluations on different splitting strategies as well deep dive into results from LLM self eval experiments
* utils.py
    * Contains helper functions to assist with pipeline development and experimentation





## Chunking Methodology

Evaluated the following methodologies



1. **Recursively split by characters (RecursiveSplitter)**
2. **Splitting by semantic context (SemanticChunker)**

We considered other methodologies for splitting as well, including HTMLSplitter, MarkdownSplitter etc. However, given the variations in the pdf format, only the above listed strategies were feasible options.

**Top chunk returned by different strategies**


<table>
  <tr>
   <td><strong>Question</strong>
   </td>
   <td><strong>SemanticChunker</strong>
   </td>
   <td><strong>RecursiveSplitter</strong>
   </td>
  </tr>
  <tr>
   <td>What was the revolution witnessed by finance in the mid to later part of the 20th century?
   </td>
   <td>Second there is a large amount of ﬁnancial data available for testing. It will be interesting to see if,
<p>
sometime in the future, ﬁnancial economists eventually replace the stylized theories of equilibrium market
<p>
dynamics with a more realistic picture of the continuing struggle of learning and adapting agents who push
<p>
markets in the direction of eﬃciency, even though they never quite reach this goal. 42
   </td>
   <td>In the mid to later part of the 20th century, ﬁnance witnessed a revolution. The advent of the eﬃcient
<p>
markets hypothesis, the capital asset pricing model, and the Black/Scholes options pricing formula put the
<p>
ﬁeld on a new, solid scientiﬁc foundation. This world was built on the assumption that asset markets were
<p>
powerful computational engines, and were able to aggregate and process the beliefs and demands of traders,
<p>
leaving in prices the full set of properly processed information currently available. At the core of asset
<p>
pricing, eﬃcient market theories give a clean and compelling picture of the world which is as appealing
<p>
to ﬁnancial economists as it is potentially unappealing to ﬁnancial practitioners.1It is interesting to note
<p>
that these foundations came with a very important computational dimension. The early availability of large
<p>
machine-readable data sets, and the computational power to analyze them, laid the critical foundation for
   </td>
  </tr>
  <tr>
   <td>Why are the financial markets appealing applications for agent-based methods?
   </td>
   <td>Unfortunately, this may also make them diﬃcult to estimate
<p>
using traditional econometric tools. Agent-based modelers should be aware of these nonlinear issues, and
<p>
take them into account when evaluating market simulations. Financial markets are an important challenge for agent-based computational modelers. Financial markets
<p>
may be one of the important early areas where agent-based methods show their worth, for two basic reasons. 41
   </td>
   <td>behavior. Several of the models covered here have already done this, and more examples of using experiments
<p>
are given in Duﬀy (2005). Finance experiments are particularly appealing since they often can be done at
<p>
time scales that are reasonable for the real data. It is more credible that you can simulate a day of trading
<p>
in the laboratory, than to simulate someone’s entire life cycle.
<p>
To summarize, ﬁnancial markets are particularly well suited for agent-based explorations. They are
<p>
large well-organized markets for trading securities which can be easily compared. Currently, the established
<p>
theoretical structure of market eﬃciency and rational expectations is being questioned. There is a long list of
<p>
empirical features that traditional approaches have not been able to match. Agent-based approaches provide
<p>
an intriguing possibility for solving some of these puzzles.21Finally, ﬁnancial markets are rich in data sets
   </td>
  </tr>
</table>



**Evaluation**

We perform a qualitative evaluation of the two splitting strategies by retrieving the top chunk returned by each strategy for a given question. 

**Observations**



* For the first question, _RecursiveSplitter returns a much more relevant chunk_ than SemanticChunker. SemanticChunker also returns the same paragraph, but its returned at the 3rd spot. So values of k&lt;3 will miss this key chunk thats necessary for answering this question
* For the second question, we observe a somewhat similar pattern. _The chunk captured by RecursiveSplitter seems more relevant than SemanticChunker_.
    * However, there is a relevant chunk in the text that was _missed by both approaches_. Possible optimal chunk : Financial markets are an important challenge for agent-based computational modelers. Financial markets may be one of the important early areas where agent-based methods show their worth, for two basic reasons. First, the area has many open questions that more standard modeling approaches have not been able to resolve. Second there is a large amount of financial data available for testing.

**Insights**



* While on paper SemanticChunker is supposed to be more attuned to the context of the text, in practice, its splits/chunks are not as conducive to QA. RecursiveSplitter is more consistent/less variance and is able to provide better contexts in most scenarios. 


## LLM Self Evaluation

**Objective**

Implement a LLM self-evaluation pipeline to provide a quantitative evaluation of the different splitting strategies and their performance

**Procedure**



* For every question, retrieve the optimal context for the question from the vector database
* Pass the retrieved context along with the question to the LLM to generate an answer for the question
* Evaluate the question-answer pair by passing the question-answer texts to the LLM and ask it to evaluate the correctness of the answer. To reduce variance, this step is repeated 3 times
* The majority decision from the previous step is used to compute the accuracy of the Q-A process

The LLM self evaluation is implemented through the Experiment class, implemented within main.py

**Results**

**Accuracy against k**

<table>
  <tr>
   <td><strong>k</strong>
   </td>
   <td><strong>Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>0.878
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>0.848
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>0.848
   </td>
  </tr>
  <tr>
   <td>4
   </td>
   <td>0.828
   </td>
  </tr>
  <tr>
   <td>5
   </td>
   <td>0.858
   </td>
  </tr>
</table>


**Accuracy against Chunk Size**

<table>
  <tr>
   <td><strong>Chunk Size</strong>
   </td>
   <td><strong>Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>500
   </td>
   <td>0.848
   </td>
  </tr>
  <tr>
   <td>1000
   </td>
   <td>0.842
   </td>
  </tr>
  <tr>
   <td>2000
   </td>
   <td>0.866
   </td>
  </tr>
</table>


**Low Performing Questions**


<table>
  <tr>
   <td><strong>Question</strong>
   </td>
   <td><strong>Avg Score</strong>
   </td>
   <td><strong>Comments</strong>
   </td>
  </tr>
  <tr>
   <td>The exchange rate is given by what formula?
   </td>
   <td>0.0
   </td>
   <td>The correct context was selected and the LLM answer was correct as well. The self eval module couldnt correctly classify the Q-A pair
   </td>
  </tr>
  <tr>
   <td>What are the standard rules of a golf game?
   </td>
   <td>0.2
   </td>
   <td>None of the docs cover the basics of golf. For the few cases where the model gave correct answer, it used internal knowledge and not the provided context
   </td>
  </tr>
  <tr>
   <td>What was the gross profit of three months ended in 2013
   </td>
   <td>0.33
   </td>
   <td>Only the settings with chunk size of 500 were able to get the correct context and subsequently the answer. <strong>All larger chunk sizes failed</strong>
   </td>
  </tr>
  <tr>
   <td>The participating schools of the Traffic Bowl Competition were from what all states in the US?
   </td>
   <td>0.4
   </td>
   <td>Opposite trend as compared to above example. <strong>Chunk sizes of 2000 had best performance</strong>. Chunk sizes of 500 couldnt get the correct answer in all iterations
   </td>
  </tr>
</table>


**Insights**



* For questions that target a very specific portion of the text, smaller chunk sizes help retrieve better splits/chunks. This is probably because with smaller chunks, the corresponding vectors are more specific and correspond more strongly to the info in the text. So for answering questions that require a specific number pull like profit in year X, population of country Y, smaller chunks work better
* For questions that are broader in nature, larger chunks/splits are more appropriate as they allow the retrieval of chunks that contain all the required info in one spot. For eg, for broad questions like what are the principles of X, what happened in France in year Y? - such broad questions benefit from context chunks of larger sizes


## Final Recommendation

Based on the findings from the experiments, the best splitting/chunking strategy would be a hybrid strategy containing the following elements : 



1. RecursiveSplitter (chunk size = 2000, chunk_overlap = 200)
2. RecursiveSplitter (chunk size = 500, chunk_overlap = 200)
3. SemanticChunker [Optional]

The final value of K would be 4

Given the insights from the evaluation of the low performing questions, **we see the advantage to having both big and small set of chunks available as it helps answer both narrow and broad questions**. 

After correcting for mistakes from the LLM self evaluation, **the value of K does not seem to have a strong relationship with performance**. This is also expected to a certain degree as the only degradation that can be expected with larger values of K is the increased difficulty for the LLM to pick the correct phrases from a larger context. However, latest LLMs are performing increasingly better in being able to pinpoint specific pieces of info from longer context. So the primary constraints for K are : 



* Context window of the LLM
* Increased cost of longer prompts


## Possible Improvements



* Use collections to differentiate document types and query only within relevant document type collection/cluster
    * For eg. We can implement a 2 stage querying process. At the first stage, we identify which topic the question belongs to : Finance, Health etc. Once we identify which topic the question belongs to, for the final selection of context, we restrict our search to only those docs that belong to the selected topic
* Utilize knowledge graphs to store vectors
    * Utilize graph-based databases like Neo4j to store vectors. Existing literature suggests improved RAG performance using graph based databases that incorporate contextual information between vectors	
* Improve data extraction from PDFs, especially for tables
    * Currently, tables are extracted purely as text and dumped into the documents. To improve the interpretation of these tables, the tabular output could be stored in a more easier-to-digest format for the LLM to be able to reason better with it
    * There are also a lot of tables with uncommon structures in the pdfs which result in a very unstructured data dump. As an alternative, more precise table reading packages like camelot can be used to extract such tables with better structure.
* Use newer OpenAI models which enforce JSON output format
    * Models like `gpt-3.5-turbo-1106 `includes an in-built response_format parameter that helps enforce JSON output format
