from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from schemas.usuario_schema import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict
from webhook_calls import trigger_webhook_tool_call, trigger_webhook_message
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Tool de exemplo
@tool
def buscar_contrato_1(cpf: str) -> str:
    """Busca a primeira parte das informações do(s) contrato(s) do cliente."""
    return "Wait for external data..."


# Tool de exemplo
@tool
def buscar_contrato_2(cpf: str) -> str:
    """Busca a segunda parte das informações do(s) contrato(s) do cliente."""
    return "Wait for external data..."

tools = [buscar_contrato_1, buscar_contrato_2]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# Tool de exemplo
@tool
def buscar_contrato_2(cpf: str) -> str:
    """Busca a segunda parte das informações do(s) contrato(s) do cliente."""
    return "Wait for external data..."

tools = [buscar_contrato_1, buscar_contrato_2]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)



# Agente principal
async def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
        Você é uma assistente prestativa da Yamaha. 
        Auxilia clientes em atraso a obter a segunda via do boleto.
        Use a ferramenta `buscar_contrato_1` e `buscar_contrato_2` ao mesmo tempo para obter as informnações do contrato
    """)

    all_messages = [system_prompt] + list(state["messages"])
    response = model.invoke(all_messages)
    state = {"messages": list(state["messages"]) + [response],
            "last_ai_message": [response],
            "last_human_message": state["last_human_message"],
            "contact": state["contact"]}
    
    print(f"******STATE CONTACT********** \n\n  {state['contact']} \n\n")
    print(f"******Response saindo do invoke:********** \n\n  {response} \n\n")
    print(f"******STATE saindo do AGENTE:********** \n\n  {state}")

    return state



#Verificar tool
async def is_tool(state: AgentState) -> AgentState:
    last_ai_message = messages_to_dict(state["last_ai_message"])[-1]
    print(f"-------------LASWT AI MSG ----------------------: {last_ai_message}")
    tool_calls = last_ai_message.get("data", {}).get("tool_calls", [])
    content = last_ai_message.get("data", {}).get("content")
    if tool_calls:
        await trigger_webhook_tool_call(contact= state["contact"], tools= tool_calls)
        return state
    else:
        await trigger_webhook_message(contact= state["contact"], message= content)
        return state

# # Condição de parada
# def is_tool(state: AgentState) -> str:
#     last_ai_message = state["last_ai_message"][-1]
#     if hasattr(last_ai_message, "tool_calls") and last_ai_message.tool_calls:
#         return "continue"
#     return "end"

# Criação do grafo LangGraph
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("is_tool", is_tool)

graph.set_entry_point("agent")

graph.add_edge("agent", "is_tool")
graph.add_edge("is_tool", END)
langgraph_app = graph.compile()