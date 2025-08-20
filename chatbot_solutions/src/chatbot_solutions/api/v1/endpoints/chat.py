from typing import List, Optional, Any

from fastapi import APIRouter, status, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict 

from models.usuario_model import UsuarioModel
from schemas.usuario_schema import MessageRequestSchema, UsuarioSchema, MessageResponseSchema, Contact, Channel
from core.deps import get_session
from graphs.graph_yamaha import langgraph_app
from graphs.help_desk_graph import app
from graphs.qualificador_leads import agent_graph_leads
router = APIRouter()

@router.post('', response_model=MessageResponseSchema, status_code=status.HTTP_202_ACCEPTED)
async def post_chat(usuario: MessageRequestSchema, db: AsyncSession = Depends(get_session)):

    if not usuario.message:
        raise HTTPException(detail='A mensagem não pode estar em branco.', status_code=status.HTTP_400_BAD_REQUEST)

    previous_message = []

    async with db as session:
        query = select(UsuarioModel).filter(UsuarioModel.phone == usuario.contact.channel.phone)
        result = await session.execute(query)
        usuario_db: UsuarioSchema = result.scalars().unique().one_or_none()
        
        if not usuario_db:
            novo_usuario = UsuarioModel(
                nome=usuario.contact.name,
                document=usuario.contact.document,
                phone=usuario.contact.channel.phone,
                email=usuario.contact.channel.email,
                messages=messages_to_dict([HumanMessage(content=usuario.message)]),
                processing= True
            )
            session.add(novo_usuario)
            await session.commit()
            await session.refresh(novo_usuario)
            usuario_db = novo_usuario
        
        else:
            if usuario_db.processing == True:
                raise HTTPException(detail='Uma mensagem está sendo processada, não é possível adicionar mais mensagens', status_code=status.HTTP_406_NOT_ACCEPTABLE)
            previous_message = messages_from_dict(usuario_db.messages)
            previous_message.append(HumanMessage(content=usuario.message))
            usuario_db.messages = messages_to_dict(previous_message)
            await session.commit()
            await session.refresh(usuario_db)

    contato = Contact(name=usuario_db.nome,
                      document=usuario_db.document,
                      channel=Channel(phone=usuario_db.phone, 
                                      email=usuario_db.email))


    state = {"messages": previous_message,
             "last_ai_message": None,
             "last_human_message":[HumanMessage(content=usuario.message)],
             "contact": contato}
        
    if usuario.project == "Yamaha Cobrança IA":
        async for step in langgraph_app.astream(state, stream_mode="values"):
            final_state = step
    if usuario.project == "HelpDesk IA":
        async for step in app.astream(state, stream_mode="values"):
            final_state = step
    if usuario.project == "Qualificador Leads IA":
        async for step in agent_graph_leads.astream(state, stream_mode="values"):
            final_state = step
    response: MessageResponseSchema = MessageResponseSchema(message=final_state["messages"][-1].content, contact= contato)

    async with db as session:
        query = select(UsuarioModel).filter(UsuarioModel.phone == usuario.contact.channel.phone)
        result = await session.execute(query)
        usuario_db: UsuarioSchema = result.scalars().unique().one_or_none()

        if usuario_db:
            usuario_db.processing = False
            usuario_db.messages = messages_to_dict(final_state["messages"])
            await session.commit()
            await session.refresh(usuario_db)
    
    return response