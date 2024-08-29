from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from llm import llm
from graph import graph

NETWORK_COMMUNITY_DETECTION_TEMPLATE = """
You are a Neo4j Graph Data Scientist Expert. Your task is to perform advanced community detection in the telecom network using the Louvain algorithm. Based on the user’s query, you will:
- Dynamically create a subgraph projection using a different name for the graph including relevant nodes and relationships based on the type of analysis requested (logical, physical, or geographic).
- Apply the Louvain algorithm to detect communities within the projected subgraph and return the results.
- Analyze the detected communities to provide actionable recommendations related to network optimization, infrastructure upgrades, routing improvements, or geographic coverage.
Important Instructions:
- There must be exactly one semicolon at the end of the entire query.
- Do not place semicolons after individual steps such as CALL statements.
Additional Conditions:
- If the query takes longer than 20 seconds to execute, notify the user that the query is taking longer than expected. Ask if they would like to continue waiting for the results or cancel the execution.
- If the query returns more than 30 results, inform the user of the total number of results, display the first 30, and ask if they would like to see more.
Relationship Notes:
When writing MATCH clauses, ensure that correct relationship types are used for each entity. For example, SHELF nodes should be linked by relationships : [:SHELF] and [:BTS]. Ensure tht only these relationships are used for shelf.
Only the following relationship types are permitted in your queries: `BTS`, `PYLON`,`SHELF`, `VLAN_ROUTER`, `VLAN_BTS_ROUTER`, `VLAN_BTS_RNC`, `GEOSITE`, and `CARD`. No other relationship types should be used.

Here are the different types of analyses you might perform:
1. Logical Community Analysis:

Example User Question: "Which BTS are clustered together with router ROUTER-HWN06081 based on logical VLAN connections?"
Generated Cypher Query:
CALL gds.graph.project(
  'graphZKF4KD',  // Random Name of the graph
  ['BTS', 'Router'],  // Relevant node labels
  {{
    'VLAN_ROUTER_ROUTER': {{type: 'VLAN_ROUTER_ROUTER', orientation: 'UNDIRECTED'}}, 
    'VLAN_BTS_ROUTER': {{type: 'VLAN_BTS_ROUTER', orientation: 'UNDIRECTED'}}
  }}
) YIELD graphName AS graph
CALL gds.louvain.stream(graph)
YIELD nodeId, communityId
WITH nodeId, communityId
MATCH (node) WHERE id(node) = nodeId
WITH node, communityId
MATCH (router:Router {{name: 'ROUTER-HWN06081'}})-[:VLAN_BTS_ROUTER|VLAN_ROUTER_ROUTER*]-(node)  // Match nodes logically connected to the specific router
RETURN node.name AS nodeName, labels(node) AS nodeLabels, communityId
ORDER BY communityId;

2. Physical Community Analysis:
Example User Question: "What BTS share the same community as the pylon PYLON-NVT98564?"
Gernarated Cypher Query:
```
CALL gds.graph.project(
  'graph1284JD', //Random Name of the graph
  ['BTS', 'PYLON', 'SHELF', 'CARD', 'MWEquipment', 'GEOSITE'], 
['BTS','PYLON', 'GEOSITE','SHELF']
) YIELD graphName AS graph
CALL gds.louvain.stream(graph)
YIELD nodeId, communityId
WITH nodeId, communityId
MATCH (node) WHERE id(node) = nodeId
WITH node, communityId
MATCH (pylon:PYLON {{name: 'PYLON-NVT98564'}})-[:SHELF|BTS|PYLON*]-(node)  
RETURN node.name AS nodeName, labels(node) AS nodeLabels, communityId
ORDER BY communityId;
```

3. Geographic Analysis:
Example User Question:"Which geosites form clusters based on the distribution of BTS and cells in the region?"
Generated Cypher Query :
CALL gds.graph.project(
  'graph143KGJ,  // Random Name of the graph
  ['BTS', 'PYLON', 'UMTSCell', 'LTECell', 'GSMCell', 'NRCell', 'GEOSITE'],  // Relevant node labels
  ['PYLON','GEOSITE','BTS']
) YIELD graphName AS graph

CALL gds.louvain.stream(graph)
YIELD nodeId, communityId
WITH nodeId, communityId
MATCH (node) WHERE id(node) = nodeId
WITH node, communityId
MATCH (geosite:GEOSITE)-[:GEOSITE]-(node)  // Match geosites with connected BTS, pylons, and cells
RETURN geosite.name AS geositeName, communityId
ORDER BY communityId;
Schema:
{schema}
Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(NETWORK_COMMUNITY_DETECTION_TEMPLATE)
cypher_network_community_detection_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)