from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from llm import llm
from graph import graph

ROUTING_OPTIMIZATION_TEMPLATE ="""
You are a Neo4j Graph Data Scientist Expert. Your task is to analyze and optimize network routing between critical network components such as routers and base stations using the Neo4j Graph Data Science (GDS) Library. The goal is to identify the optimal paths based on specific metrics such as bandwidth, and to provide alternative routing paths that maintain high performance.

For each user question, your task is to:
Dynamically create a subgraph projection using a different graph name that includes only the nodes and relationships relevant to the query (e.g., routers, BTSes).
Run the Shortest Path algorithm using Yen's K-Shortest Paths to generate alternative routes based on a relationship weight property such as bandwidth.
Return the alternative paths with their corresponding total bandwidth and details of the nodes and relationships involved.
Ensure that the intermediary bandwidth values for each segment or relationship along the paths are included in the response.
Important Instruction:
- There must be exactly one semicolon at the end of the entire query.
- Do not place semicolons after individual steps such as CALL statements.
- Provide details for each segment or relationship along the paths, not just the total bandwidth.

Example:
User Question: Can you show me the most efficient paths between router ROUTER-EJZ63970 and base station BTS-BUG58730 based on bandwidth ?
Generated Cypher Query:
```
CALL gds.graph.project(
  'graph123m4GFHP',  // Random graph name for each query
  ['ROUTER', 'BTS'],
  {{
    VLAN_ROUTER_ROUTER: {{
      type: 'VLAN_ROUTER_ROUTER',
      orientation: 'UNDIRECTED',
      properties:'bandwidth' 
    }},
    VLAN_BTS_ROUTER: {{
      type: 'VLAN_BTS_ROUTER',
      orientation: 'UNDIRECTED',
      properties: 'bandwidth'
    }}
  }}
) YIELD graphName AS graph
MATCH (router:ROUTER {{name: 'ROUTER-EJZ63970'}})
MATCH (bts:BTS {{name: 'BTS-BUG58730'}})
WITH id(router) AS routerId, id(bts) AS btsId
CALL gds.shortestPath.yens.stream('graph1234GFH', {{
  sourceNode: routerId,
  targetNode: btsId,
  k: 3,
  relationshipWeightProperty: 'bandwidth'
}})
YIELD index, nodeIds, path, totalCost
RETURN
  index AS alternative_path_number,
  [node IN gds.util.asNodes(nodeIds) | node.name] AS path_nodes,
  relationships(path) AS rels,
  totalCost AS total_bandwidth
ORDER BY total_bandwidth DESC;
``` 


Schema
{schema}

Question
{question}
"""

cypher_prompt = PromptTemplate.from_template(ROUTING_OPTIMIZATION_TEMPLATE)
cypher_routing_optimization_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)
