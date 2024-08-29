from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from tools.general_cypher_queries import cypher_telecom_qa
from tools.critical_nodes_analysis import cypher_critical_nodes_qa
from tools.routing_optimization import cypher_routing_optimization_qa
from tools.network_community_detection import cypher_network_community_detection_qa

from utils import get_session_id

chat_prompt = ChatPromptTemplate.from_messages(
     [
        ("system", (
              "You are a highly skilled telecommunications network expert specialized in graph topology analysis. "
            "Your role is to provide detailed explanations and insights related to telecommunications networks, including equipment, topology, and connectivity. "
            "You do not generate or translate user queries into Cypher queries, nor do you engage in Neo4j database manipulation. "
            "Instead, you offer in-depth understanding and guidance on network architecture, best practices, and solutions to general network challenges."
            
        )),
        ("human", "{input}"),
    ]
)

network_chat = chat_prompt | llm | StrOutputParser()
tools = [
    Tool.from_function(
        name="Telecom Network Domain Expert",
        description="Use this tool for in-depth discussions and expert advice on general telecommunications network concepts, architecture, and best practices. Ideal for broad questions or advice on telecom network design and management.",
        func=network_chat.invoke,
    ),
    Tool.from_function(
        name="Cypher Query for Telecom Topology",
        description="Use this tool to generate Cypher queries related to telecom network topology, such as retrieving the shelf of a port or the card of a BTS. This tool ensures accurate retrieval of network components and their relationships using Neo4j.",
        func=cypher_telecom_qa.invoke,
    ),
    Tool.from_function(
        name="Critical Network Points Analysis",
        description="Use this tool to analyze critical points in the telecom network topology. It identifies key network nodes and connections that are vital for network performance and reliability.",
        func=cypher_critical_nodes_qa
    ),
    Tool.from_function(
        name="Routing Path Optimization",
        description="Use this tool to find the most efficient routing paths between network components like routers and base stations. It applies graph algorithms to identify the shortest and most efficient paths based on metrics such as bandwidth.",
        func=cypher_routing_optimization_qa
    ),
    Tool.from_function(
        name="Network Community Detection",
        description="Use this tool for advanced community detection in telecom networks using the Louvain algorithm. It helps in generating subgraphs based on logical, physical, or geographic criteria, and provides recommendations for network optimization and infrastructure improvements.",
        func=cypher_network_community_detection_qa
    )
]


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a telecommunications network expert providing information specialized in graph topology analysis.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to telecommunications network graph topology analysis.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']