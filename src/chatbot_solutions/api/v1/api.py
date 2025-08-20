from fastapi import APIRouter

from api.v1.endpoints import chat, submit_tools


api_router = APIRouter()

api_router.include_router(chat.router, prefix = '/chat', tags=['chat'])
api_router.include_router(submit_tools.router, prefix = '/submit_tools', tags=['submit_tools'])