from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from config import OPENAI_API_KEY, SYSTEM_PROMPT
from tools import tools

class InventoryAgent:
    def __init__(self):
        # Initialize the LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("llm_decision_step", self._llm_decision_step)
        builder.add_node("tools", ToolNode(tools))
        
        # Add edges
        builder.add_edge(START, "llm_decision_step")
        builder.add_conditional_edges(
            "llm_decision_step",
            tools_condition,
        )
        builder.add_edge("tools", "llm_decision_step")
        
        return builder.compile()
    
    def _llm_decision_step(self, state: MessagesState):
        """LLM decision step function"""
        user_question = state["messages"]
        input_question = [SystemMessage(content=SYSTEM_PROMPT)] + user_question
        response = self.llm_with_tools.invoke(input_question)
        return {"messages": [response]}
    
    def process_query(self, query: str):
        """Process a user query and return the response"""
        message = [HumanMessage(content=query)]
        result = self.graph.invoke({"messages": message})
        return result["messages"][-1].content
    
    def get_graph_visualization(self):
        """Get the graph visualization (returns bytes for PNG)"""
        try:
            return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            print(f"Error generating graph visualization: {e}")
            return None

# Initialize the agent
agent = InventoryAgent()
