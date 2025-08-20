from typing import List, Optional, Any

from fastapi import APIRouter, status, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,ToolMessage, messages_from_dict, messages_to_dict 

from models.usuario_model import UsuarioModel
from schemas.usuario_schema import ToolCallRequestSchema, UsuarioSchema, MessageResponseSchema, Contact, Channel
from core.deps import get_session
from graphs.graph_yamaha import langgraph_app

router = APIRouter()

@router.post('', response_model=MessageResponseSchema, status_code=status.HTTP_202_ACCEPTED)
async def post_chat(tool_calls_response: ToolCallRequestSchema, db: AsyncSession = Depends(get_session)):

    if not tool_calls_response.tool_calls:
        raise HTTPException(detail="Missing tool calls.", status_code=status.HTTP_400_BAD_REQUEST)
    
    tool_response = []
    
    for tool_call in tool_calls_response.tool_calls:
        tool_msg = ToolMessage(tool_call_id=tool_call.tool_call_id, 
                               content= tool_call.content)
        tool_response.append(tool_msg)

    

    async with db as session:
        query = select(UsuarioModel).filter(UsuarioModel.phone == tool_calls_response.contact.channel.phone)
        result = await session.execute(query)
        usuario_db: UsuarioSchema = result.scalars().unique().one_or_none()

        # Garante que usuario_db.messages seja uma lista válida
        previous_message = messages_from_dict(usuario_db.messages or [])

        # Garante que tool_msg está definido corretamente
        for tool_call in tool_calls_response.tool_calls:
            tool_msg = ToolMessage(tool_call_id=tool_call.tool_call_id, content=tool_call.content)
            previous_message.append(tool_msg)

        # Converte de volta para dict
        usuario_db.messages = messages_to_dict(previous_message)
        await session.commit()
        await session.refresh(usuario_db)
    


    contato = Contact(name=usuario_db.nome,
                      document=usuario_db.document,
                      channel=Channel(phone=usuario_db.phone, 
                                      email=usuario_db.email))
    
    print(previous_message)

    state = {"messages": previous_message,
             "last_ai_message": None,
             "last_human_message": None,
             "contact": contato}
        
    async for step in langgraph_app.astream(state, stream_mode="values"):
        final_state = step

    response: MessageResponseSchema = MessageResponseSchema(message=final_state["messages"][-1].content, contact= contato)

    async with db as session:
        query = select(UsuarioModel).filter(UsuarioModel.phone == tool_calls_response.contact.channel.phone)
        result = await session.execute(query)
        usuario_db: UsuarioSchema = result.scalars().unique().one_or_none()

        if usuario_db:
            usuario_db.messages = messages_to_dict(final_state["messages"])
            await session.commit()
            await session.refresh(usuario_db)
    
    return response