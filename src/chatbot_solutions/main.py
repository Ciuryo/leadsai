from fastapi import FastAPI

from core.configs import settings
from api.v1.api import api_router
import logging


logging.basicConfig(
    level=logging.INFO,  # ou DEBUG se quiser mais detalhes
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title='Chat API - IA')
app.include_router(api_router, prefix=settings.API_V1_STR)



if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level='info')
