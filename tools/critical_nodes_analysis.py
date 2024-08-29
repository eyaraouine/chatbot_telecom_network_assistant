from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from llm import llm
from graph import graph

CRITICAL_POINTS_ANALYSIS_TEMPLATE = """
You are a Neo4j Graph Data Scientist Expert. Your task is to analyze the critical points within a graph topology by using the Neo4j Graph Data Science (GDS) Library. The objective is to identify the most critical nodes that play a crucial role in the network based on PageRank.
For each user question, your task is to:
Dynamically create a subgraph projection using a different name for the graph, including only the nodes and relationships relevant to the specific query (e.g., routers, BTSes, RNCs, etc.).
Run the PageRank algorithm to rank the importance of nodes within the network based on their connections.
Return and interpret the top-ranked nodes with their scores, explaining their significance in the network.
Important Instructions:
- There must be exactly one semicolon at the end of the entire query.
- Do not place semicolons after individual steps such as CALL statements.

Here are some queries examples based on different node types:

Example 1: Router Critically Analysis
User Question :"Analyse the 10 most critical routers in the network" 
Generated Cypher Query:
```
CALL gds.graph.project(
  'graph1234SHFH',
  ['ROUTER', 'BTS'],
  ['VLAN_ROUTER_ROUTER', 'VLAN_BTS_ROUTER']
) YIELD graphName AS graph
CALL gds.pageRank.stream(graph, {{
  maxIterations: 20,
  concurrency: 4
}}) YIELD nodeId, score
WITH nodeId, score
MATCH (router:ROUTER) WHERE id(router) = nodeId
RETURN router.name AS routerName, score
ORDER BY score DESC LIMIT 10;
```
Example 2: BTS Critically Analysis
User Question:"Which BTS has the highest criticality score in the current network topology?"
Generated Cypher Query:
```
CALL gds.graph.project(
  'Graphkdks5GJ',
  ['BTS','SHELF','UMTSCell','NBIOTCell','NRCell','GSMCell','LTECell'],
  'BTS'
) YIELD graphName AS graph
CALL gds.pageRank.stream(graph, {{
  maxIterations: 20,
  concurrency: 4
}}) YIELD nodeId, score
WITH nodeId, score
MATCH (bts:BTS) WHERE id(bts) = nodeId
RETURN bts.name AS btsName, score
ORDER BY score DESC LIMIT 1;
```
Schema
{schema}

Question
{question}
"""

cypher_prompt = PromptTemplate.from_template(CRITICAL_POINTS_ANALYSIS_TEMPLATE)
cypher_critical_nodes_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)
