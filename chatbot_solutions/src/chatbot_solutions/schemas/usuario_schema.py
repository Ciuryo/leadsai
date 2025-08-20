from typing import Optional
from typing import List , Dict, Any
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, EmailStr

class Channel(BaseModel):
    phone: str
    email: Optional[EmailStr]

class Contact(BaseModel):
    name: str
    document: str
    channel: Channel

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    last_ai_message: AIMessage
    last_human_message: HumanMessage
    contact: Contact


class ToolCallSchema(BaseModel):
    tool_call_id: str
    content: str

class ToolCallResponseSchema(BaseModel):
    tool_calls: List[ToolCallSchema]

class ToolCallRequestSchema(BaseModel):
    tool_calls: List[ToolCallSchema]
    contact: Contact

class MessageResponseSchema(BaseModel):
    message: str
    contact: Contact

class MessageRequestSchema(BaseModel):
    message: str
    project: str
    contact: Contact

class UsuarioSchema(BaseModel):
    id: int
    nome: str
    document: str
    phone: str
    email: EmailStr
    messages: List[Dict[str, Any]]
    class Config:
        orm_mode=True