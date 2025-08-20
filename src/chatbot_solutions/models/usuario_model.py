from sqlalchemy import Integer, String, Column, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from core.configs import settings

class UsuarioModel(settings.DBBaseModel):
    __tablename__ = 'usuarios'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nome = Column(String(256), nullable=True)
    document = Column(String(256), nullable=True)
    phone = Column(String(256), index=True, nullable=False, unique=False)
    email = Column(String(256), index=True, nullable=True, unique=False)
    messages = Column(JSONB, nullable=False)
    processing = Column(Boolean, nullable=True)
