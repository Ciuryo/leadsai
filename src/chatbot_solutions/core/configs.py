from typing import List
from pydantic_settings import BaseSettings
from sqlalchemy.orm import declarative_base
from typing import ClassVar

Base = declarative_base()

class Settings(BaseSettings):
    API_V1_STR: str = '/api/v1'
    DB_URL: str = "postgresql+asyncpg://postgres:123@localhost:5432/usuarios"
    DBBaseModel: ClassVar = Base

    JWT_SECRET: str = 'PjfPIWN_kbCJHEfHNA40oqq3-WJhZLHfKtv0iQPGpxY'
    """
    import secrets

    token: str = secrets.token_urlsafe(32)
    """
    ALGORITHM: str = 'HS256'
    # 60 minutos * 24 horas * 7 dias => 1 semana
    ACESS_TOKEN_EXPIRE_MINUTES: int = 60*24*7

    class Config:
        case_sensitive = True


settings: Settings = Settings()