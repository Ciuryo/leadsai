import uuid
import httpx
from schemas.usuario_schema import Contact
import logging

WEBHOOK_URL = "https://i-dr.io/v1/idr/webhook/api/B20FE8F073FAFA1F/fe6bbe5a-662c-433c-b826-01d2111a0db2"

logger = logging.getLogger(__name__)

async def trigger_webhook_tool_call(contact: Contact, tools: dict):
    payload = {
    "data": tools,
    "contact": contact.dict()
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(WEBHOOK_URL, json=payload)
            response.raise_for_status()
            logger.info("Webhook enviado com sucesso. Status: %s", response.status_code)
    except httpx.HTTPStatusError as e:
        logger.error("Erro HTTP ao enviar webhook: %s - %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Erro de requisição ao enviar webhook: %s", str(e))
    except Exception as e:
        logger.exception("Erro inesperado ao enviar webhook: %s", str(e))


async def trigger_webhook_message(contact: Contact, message: str):
    payload = {
    "data": message,
    "contact": contact.dict()
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(WEBHOOK_URL, json=payload)
            response.raise_for_status()
            logger.info("Webhook enviado com sucesso. Status: %s", response.status_code)
    except httpx.HTTPStatusError as e:
        logger.error("Erro HTTP ao enviar webhook: %s - %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Erro de requisição ao enviar webhook: %s", str(e))
    except Exception as e:
        logger.exception("Erro inesperado ao enviar webhook: %s", str(e))
