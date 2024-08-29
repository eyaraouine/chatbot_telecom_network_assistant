from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from llm import llm
from graph import graph

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer. Your task is to translate user questions into Cypher queries without using Neo4J GDS Library. Always ensure the correct matching of nodes and relationships before attempting to return them.
Important Instructions:
- Use only the provided relationship types and properties in the schema.
- Only these relationships types exist: BTS,SHELF, CARD, VLAN_ROUTER_ROUTER, VLAN_BTS_ROUTER, GEOSITE, PYLON. Do not use any other types.
- If a generated cypher query successfully returns results, do not attempt additional queries. Stop once a valid result is obtained.
- Before generating a new Cypher query, check if the user’s question matches any of the examples provided below. If a match is found, adapt the example query to fit the user’s details. If no match is found, create a new query based on the user’s question.
- If the Cypher query returns results, stop any further queries or chains and use the result directly.

Examples to follow:
User Question: "What is the card that contains the port PORT-ADD45712?"
Generated Cypher: 
```
MATCH (c:CARD)<-[:CARD]-(p:PORT {{name: 'PORT-ADD45712'}}) RETURN c.name
```
User Question: "What are the cells asociated to BTS BTS-FZS86613?"
Generated Cypher: 
```
MATCH (m:UMTSCell|NBIOTCell|LTECell|NRCell|GSMCell )-[r:BTS]->(p:BTS{{name:'BTS-FZS86613'}}) RETURN m.name
```
User Question: "what are the nodes that are directly or indirectly connected to port PORT-JCD18072 with a depth limit of 2?"
Generated Cypher: 
```
MATCH (p:PORT {{name: 'PORT-JCD18072'}})-[*1..2]-(n) RETURN n.name;
```
User Question: "What is the shelf of the port PORT-VLM92169?"
Generated Cypher: 
```
MATCH (p:PORT {{name: 'PORT-VLM92169'}})-[:CARD]->(c:CARD)-[:SHELF]->(s:SHELF) RETURN s.name
```

User Question: "what is the bts of the port PORT-EQD28999?"
Generated Cypher: 
```
MATCH (p:PORT {{name: 'PORT-EQD28999'}})-[:CARD]->(c:CARD)-[:SHELF]->(s:SHELF)-[:BTS]->(b:BTS) RETURN b.name;
```

Schema:
{schema}

Question:
{question}

"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
cypher_telecom_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)