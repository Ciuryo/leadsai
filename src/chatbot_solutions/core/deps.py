from typing import Generator, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

from core.database import Session


class TokenData(BaseModel):
    username: Optional[str] = None


async def get_session() -> Generator:
    session: AsyncSession = Session()

    try:
        yield session
    finally:
        await session.close()
