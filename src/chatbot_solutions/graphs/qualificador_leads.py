import os
import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from schemas.usuario_schema import AgentState
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, Timeout, RequestException

# ETAPA 1: CONFIGURAÇÕES E CONHECIMENTO BASE
#edit
# Carrega variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Chave da API da OpenAI não encontrada. Verifique seu arquivo .env")

# Token do RD Station
RD_TOKEN = os.getenv("RD_TOKEN").strip()

# Define o LLM principal usado por todos os agentes da CrewAI e o agente principal
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=API_KEY,
    temperature=0.8
)

# Este é o conhecimento base "embutido" no agente.
ROBBU_KNOWLEDGE_BASE = """
Nossa história: Fundada em 2016, a Robbu é líder em soluções de automação de comunicação entre marcas e seus clientes.
A Robbu é uma empresa especializada em soluções digitais para comunicação, automação e atendimento omnichannel. Atua no desenvolvimento de plataformas e APIs que integram canais como WhatsApp, voz, e-mail, SMS, redes sociais e webchat, com foco em automação inteligente, eficiência operacional e escalabilidade para médias e grandes empresas. A Robbu é reconhecida pela robustez de suas soluções, integração com inteligência artificial e experiência comprovada em projetos de grande porte. 

Sinergia entre tecnologia e conexões:
Nascemos com a crença de que o sucesso de uma empresa está na comunicação e no relacionamento personalizado que ela possui com seus clientes.
Ao combinar nosso DNA de customer experience com o que há de melhor em atendimento e vendas no mundo digital, criamos a Robbu,
uma solução inteligente que transforma o contato de marcas com seus clientes.

Parcerias e Alcance
A empresa é parceira do Google, Meta e Microsoft, provedora oficial do WhatsApp Business API e foi a Empresa Destaque do Facebook (Meta) em 2021.
Com sede em São Paulo, Brasil, e presente em outros três países: Argentina, Portugal e Estados Unidos, a Robbu conta com mais de 800 clientes e parceiros de negócios em 26 países.

Invenio
Descrição: 
Plataforma omnichannel de atendimento digital, vendas e gestão, integrando múltiplos canais (WhatsApp, voz, e-mail, SMS, redes sociais, webchat) em uma única interface. Centraliza todas as conversas do cliente, independentemente do canal ou momento, proporcionando uma jornada única e contínua. 

WhatsApp Business Calling API
Descrição: 
API para chamadas de voz bidirecionais via WhatsApp, permitindo que empresas e usuários iniciem ligações diretamente pelo aplicativo, com integração nativa ao Invenio. 

Voice by Robbu 
Descrição: 
Central de atendimento inteligente baseada em voz, com telefonia em nuvem, discador próprio, integração ao WhatsApp e relatórios com inteligência artificial. Permite a transição fluida entre atendimento por telefone e WhatsApp, mantendo o histórico do cliente para uma experiência contínua. 

Robbu Verify
Descrição: 
Ferramenta para validação, higienização e classificação de bases de contatos telefônicos. 

Maestro:
Descrição: 
Orquestrador e padronizador de regras de negócio, centralizando a criação, revisão, aprovação e envio de templates de mensagens para múltiplos canais (exceto voz). Permite colaboração entre diferentes times e empresas, com controle total do fluxo operacional. 

Nossos Produtos e Soluções:
Invenio Center: Gestão operacional e estratégica, com dashboards em tempo real, relatórios detalhados, criação de campanhas digitais integradas, controle de performance e definição de estratégias assertivas.
- Carteiro Digital: Disparo de comunicação em massa (campanhas, notificações e pesquisas) via WhatsApp, SMS e outros canais.
- APIs de Comunicação: APIs robustas (ex.: WhatsApp Business) para integrar funcionalidades de comunicação aos sistemas dos clientes.
- Webchat: Chat personalizável para sites, integrado à nossa plataforma.
- rFlow: Bloqueador de chamadas que garante qualidade no contato ativo com os clientes.
Invenio Live: Atendimento unificado em todos os canais digitais, com recursos como fila inteligente, distribuição automática preditiva, frases prontas, segmentação de contatos e integração com sistemas externos. 
- chatbots com IA: Soluções de automação de atendimento com inteligência artificial, capazes de aprender e evoluir com o tempo.
-Segurança:
A Robbu segue corretamente as normas da Lei Geral de Proteção de Dados (LGPD).

Case de sucesso da Robbu:
- Yamaha: Invenio + chatbot → +500% atendimentos, 40% resolvidos sem humano, NPS elevado.
- Tibério Construtora: Click to WhatsApp → +R$4 milhões em vendas no 1º tri.
- STF: Canal oficial WhatsApp → combate a fake news, até 100 mil pessoas/dia atendidas.
- Paschoalotto: Invenio + cobrança digital → 30% da carteira recuperada só com chatbot.
- Ministério da Saúde: Chatbot WhatsApp → 50% atendimentos resolvidos, -30% erros cadastrais.
- Liberty Seguros: Automação no Invenio → 50% dos casos resolvidos direto, triplo de atendimentos.
- Defesa Civil: Alertas via WhatsApp → 19 mil alertas em 2024, pioneirismo mundial.
"""

# Essas informações são usadas pela CrewAI quando ela é acionada.
ROBBU_DOCS_CONTEXT = [
    {"name": "Carteiro Digital", "url": "https://docs.robbu.global/docs/carteiro-digital/carteiro-digital"},
    {"name": "Webhook", "url": "https://docs.robbu.global/docs/center/webhook"},
    {"name": "Webchat", "url": "https://docs.robbu.global/docs/center/web-chat"},
    {"name": "Invênio", "url": "https://robbu.global/produtos/invenio-center/"},
    {"name": "Invenio live", "url": "https://robbu.global/produtos/invenio-live/"},
    {"name": "IDR Studio", "url": "https://robbu.global/produtos/idr-chatbot-studio/"},
    {"name": "Invenio Webchat", "url": "https://robbu.global/produtos/webchat/"},
    {"name": "Positus WhatsApp", "url": "https://robbu.global/produtos/whatsapp-studio-positus/"},
    {"name": "rFlow", "url": "https://robbu.global/produtos/rflow/"},
    {"name": "Campanha email", "url": "https://docs.robbu.global/docs/center/campanhas-de-email"},
    {"name": "Campanha whats app", "url": "https://docs.robbu.global/docs/center/campanhas-de-whatsapp"},
]

# ESTADO DO LEAD (memória de sessão)

LEAD_STATE: Dict[str, Optional[str]] = {
    "emailLead": None,
    "nomeLead": None,
    "siteEmpresa": None,
    "cargoCliente": None,
    "numeroCliente": None,
    "idContato": None,
    "status_lead": None,          # "quente" | "frio"
    "interesse_produto": None,    # Chatbot IA, Invênio, Carteiro Digital, Campanhas, Outros
    "tamanho_time": None,         # Número (string)
    "segmento": None,             # Segmento informado
}

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def _domain_like(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return bool(re.match(r"^https?://", s) or re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", s))

# ETAPA 2: FERRAMENTAS - CREW COMO FERRAMENTA (mantido)

class ContextSearchTool(BaseTool):
    name: str = "Busca na Documentação Interna"
    description: str = "Busca a URL da documentação mais relevante para uma pergunta técnica."

    def _find_best_match(self, query: str, docs_context: List[Dict]) -> Tuple[Dict, int]:
        query_lower = query.lower()
        best_match_doc = None
        best_score = 0
        query_tokens = set(re.findall(r"\w+", query_lower))
        for doc in docs_context:
            score = 0
            doc_name_lower = doc["name"].lower()
            doc_keywords = {k.lower() for k in doc.get("keywords", [])}
            if any(tok in doc_name_lower for tok in query_tokens):
                score += 5
            score += len(query_tokens.intersection(doc_keywords)) * 3
            if score > best_score:
                best_score = score
                best_match_doc = doc
        return best_match_doc, best_score

    def _run(self, query: str) -> str:
        try:
            robbu_match, robbu_score = self._find_best_match(query, ROBBU_DOCS_CONTEXT)
        except Exception as e:
            return f"[ERRO_BUSCA_DOCS:{str(e)}]"

class EnhancedWebScrapeTool(BaseTool):
    name: str = "Extração Avançada de Conteúdo Web"
    description: str = "Extrai conteúdo de páginas web com tratamento de erros e formatação."

    def _run(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            main_content = (soup.find('main') or soup.find('article') or soup.find('body'))
            text = main_content.get_text("\n", strip=True) if main_content else ""
            return '\n'.join([line.strip() for line in text.split('\n') if line.strip()])[:4000]
        except Exception as e:
            return f"[ERRO_EXTRACAO:{str(e)}]"

# - A LÓGICA DA CREW ENCAPSULADA EM UMA CLASSE -
class TechnicalCrewExecutor:
    def run(self, query: str) -> str:
        """Executa a Crew de agentes para encontrar uma resposta técnica."""
        pesquisador = Agent(
            role="Especialista em Pesquisa de Documentação",
            goal="Localizar a URL mais relevante na base de conhecimento para responder a uma pergunta técnica.",
            backstory="Você é um especialista em encontrar a fonte de informação correta dentro de uma base de conhecimento pré-definida para responder a pergunta do usuário.",
            llm=llm, tools=[ContextSearchTool()], verbose=True
        )
        extrator = Agent(
            role="Extrator de Conteúdo Web",
            goal="Extrair o conteúdo essencial de uma página web de forma limpa e objetiva.",
            backstory="Especialista em parsing de HTML, focado em extrair apenas o texto relevante de uma página.",
            llm=llm, tools=[EnhancedWebScrapeTool()], verbose=True
        )
        redator = Agent(
            role="Redator Técnico Profissional",
            goal="Produzir respostas claras, objetivas e profissionais baseadas em conteúdo técnico. Você transforma jargão técnico em respostas fáceis de entender formatadas de forma personalizada para o usuário final.",
            backstory="Você é um especialista em suporte técnico que se comunica de forma concisa e direta, sempre citando a fonte oficial.",
            llm=llm, verbose=True
        )

        tarefa_pesquisa = Task(description=f"Encontre a URL da documentação para: '{query}'", expected_output="A URL mais relevante.", agent=pesquisador)

        crew_pesquisa = Crew(agents=[pesquisador], tasks=[tarefa_pesquisa], process=Process.sequential, verbose=True, telemetry=False)
        url = crew_pesquisa.kickoff()

        if not str(url).startswith("http"):
            return "Não localizei uma página específica para essa dúvida. Você pode detalhar o erro, código ou endpoint?"

        tarefa_extracao = Task(description=f"Extraia o conteúdo da URL: {url}", expected_output="Texto limpo e objetivo.", agent=extrator)
        tarefa_redacao = Task(
            description=f"Produza uma resposta técnica objetiva para a pergunta '{query}', usando o conteúdo extraído. Sua resposta deve ser em português, profissional, e terminar com a fonte: {url}",
            expected_output="Resposta técnica profissional.",
            agent=redator,
            context=[tarefa_extracao]
        )

        crew_completa = Crew(agents=[extrator, redator], tasks=[tarefa_extracao, tarefa_redacao], process=Process.sequential, verbose=True, telemetry=False)
        resultado_final = crew_completa.kickoff()

        return resultado_final if resultado_final else f"Não foi possível processar a solicitação para a URL: {url}"

@tool
def pesquisa_tecnica_avancada_robbu(query: str) -> str:
    """
    Use esta ferramenta para responder a perguntas técnicas específicas sobre a plataforma Robbu, como os produtos da robbu.
    """
    executor = TechnicalCrewExecutor()
    return executor.run(query)

# ETAPA 2.1: NOVAS FERRAMENTAS (Lead, RD Station, Roteamento)

@tool
def salvar_dado_lead(campo: str, valor: str) -> str:
    """
    Salva um dado do lead na memória de sessão e sincroniza imediatamente com o RD Station.
    Campos previstos: emailLead, nomeLead, siteEmpresa, cargoCliente, numeroCliente, interesse_produto, tamanho_time, segmento.
    Retorna o estado atual do lead em JSON, incluindo o resultado da operação no RD Station.
    """
    campo = campo.strip()
    LEAD_STATE[campo] = (valor or "").strip() or None

    rd_result = None
    try:
        # Só tenta sincronizar se já houver emailLead preenchido
        email = LEAD_STATE.get("emailLead", "") or ""
        if email:
            if LEAD_STATE.get("idContato"):
                # Atualiza contato existente
                rd_result = rd_atualizar_contato.invoke({
                    "idContato": LEAD_STATE["idContato"],
                    "nomeLead": LEAD_STATE.get("nomeLead", ""),
                    "emailLead": email,
                    "numeroCliente": LEAD_STATE.get("numeroCliente", ""),
                    "cargoCliente": LEAD_STATE.get("cargoCliente", "")
                })
            else:
                # Cria novo contato
                rd_result = rd_criar_contato.invoke({
                    "nomeLead": LEAD_STATE.get("nomeLead", ""),
                    "emailLead": email,
                    "numeroCliente": LEAD_STATE.get("numeroCliente", ""),
                    "cargoCliente": LEAD_STATE.get("cargoCliente", "")
                })
    except Exception as e:
        rd_result = f"[ERRO_RD_STATION:{str(e)}]"

    return _safe_json({
        "ok": True,
        "atualizado": {campo: LEAD_STATE[campo]},
        "lead": LEAD_STATE,
        "rd_station_result": rd_result
    })

@tool
def listar_dados_lead() -> str:
    """Retorna o estado atual do lead em JSON."""
    return _safe_json({"lead": LEAD_STATE})

@tool
def avaliar_lead_quente() -> str:
    """
    Aplica critérios:
      - Empresa com mais de 5 funcionários (tamanho_time > 5 numérico)
      - Possui site ativo (siteEmpresa com formato de domínio/URL)
      - Interesse real (interesse_produto definido)
    Define LEAD_STATE['status_lead'] como 'quente' ou 'frio' (se faltar algum critério).
    """
    try:
        funcionarios = None
        if LEAD_STATE.get("tamanho_time"):
            m = re.search(r"\d+", LEAD_STATE["tamanho_time"])
            if m:
                funcionarios = int(m.group(0))
        tem_site = _domain_like(LEAD_STATE.get("siteEmpresa"))
        tem_interesse = bool(LEAD_STATE.get("interesse_produto"))

        quente = (funcionarios is not None and funcionarios > 5) and tem_site and tem_interesse
        LEAD_STATE["status_lead"] = "quente" if quente else "frio"
        return _safe_json({"ok": True, "criterios": {"funcionarios": funcionarios, "tem_site": tem_site, "tem_interesse": tem_interesse}, "status_lead": LEAD_STATE["status_lead"]})
    except Exception as e:
        return _safe_json({"ok": False, "erro": str(e)})

# RD Station CRM

BASE_RD = "https://crm.rdstation.com/api/v1/contacts"

def _rd_headers() -> Dict[str, str]:
    return {"accept": "application/json", "content-type": "application/json"}

@tool
def rd_listar_contatos(emailLead: str = "") -> str:
    """
    Lista contatos no RD Station por e-mail.
    GET {BASE_RD}?email={{emailLead}}&token={{RD_TOKEN}}
    Atualiza LEAD_STATE['idContato'] se encontrar um contato.
    """
    email = (emailLead or LEAD_STATE.get("emailLead") or "").strip()
    if not email:
        return _safe_json({"ok": False, "erro": "emailLead ausente"})
    url = f"{BASE_RD}?email={requests.utils.quote(email)}&token={RD_TOKEN}"
    try:
        resp = requests.get(url, headers=_rd_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Tenta identificar id (formatos comuns no RD)
        found_id = None
        if isinstance(data, dict):
            # pode trazer lista em 'contacts' ou um único contato
            if "contacts" in data and isinstance(data["contacts"], list) and data["contacts"]:
                found_id = data["contacts"][0].get("id") or data["contacts"][0].get("_id")
            else:
                found_id = data.get("id") or data.get("_id")
        elif isinstance(data, list) and data:
            found_id = data[0].get("id") or data[0].get("_id")
        if found_id:
            LEAD_STATE["idContato"] = str(found_id)
        return _safe_json({"ok": True, "url": url, "result": data, "idContato": LEAD_STATE["idContato"]})
    except (Timeout, HTTPError, RequestException) as e:
        return _safe_json({"ok": False, "erro": str(e), "url": url})

@tool
def rd_criar_contato(nomeLead: str = "", emailLead: str = "", numeroCliente: str = "", cargoCliente: str = "") -> str:
    """
    Cria contato no RD Station.
    POST {BASE_RD}?token={{RD_TOKEN}}
    Body conforme exemplo fornecido.
    Atualiza LEAD_STATE['idContato'].
    """
    nome = (nomeLead or LEAD_STATE.get("nomeLead") or "").strip()
    email = (emailLead or LEAD_STATE.get("emailLead") or "").strip()
    cargo = (cargoCliente or LEAD_STATE.get("cargoCliente") or "").strip()
    numero = (numeroCliente or LEAD_STATE.get("numeroCliente") or "").strip()

    if not email:
        return _safe_json({"ok": False, "erro": "emailLead ausente para criação"})
    payload = {
        "contact": {
            "name": nome or None,
            "emails": [{"email": email}],
            "phones": [{"type": "home", "phone": numero}] if numero else [],
            "title": cargo or None
        }
    }
    url = f"{BASE_RD}?token={RD_TOKEN}"
    try:
        resp = requests.post(url, headers=_rd_headers(), data=json.dumps(payload), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # captura id
        new_id = data.get("id") or data.get("_id")
        if not new_id and isinstance(data, dict) and "contact" in data:
            new_id = data["contact"].get("id") or data["contact"].get("_id")
        if new_id:
            LEAD_STATE["idContato"] = str(new_id)
        return _safe_json({"ok": True, "url": url, "payload": payload, "result": data, "idContato": LEAD_STATE["idContato"]})
    except (Timeout, HTTPError, RequestException) as e:
        return _safe_json({"ok": False, "erro": str(e), "url": url, "payload": payload})

@tool
def rd_exibir_contato(idContato: str = "") -> str:
    """
    Exibe contato por ID no RD Station.
    GET {BASE_RD}/{{idContato}}?token={{RD_TOKEN}}
    """
    cid = (idContato or LEAD_STATE.get("idContato") or "").strip()
    if not cid:
        return _safe_json({"ok": False, "erro": "idContato ausente"})
    url = f"{BASE_RD}/{cid}?token={RD_TOKEN}"
    try:
        resp = requests.get(url, headers=_rd_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return _safe_json({"ok": True, "url": url, "result": data})
    except (Timeout, HTTPError, RequestException) as e:
        return _safe_json({"ok": False, "erro": str(e), "url": url})

@tool
def rd_atualizar_contato(idContato: str = "", tokenRD: str = "", nomeLead: str = "", emailLead: str = "", numeroCliente: str = "", cargoCliente: str = "") -> str:
    """
    Atualiza contato no RD Station (PUT).
    PUT {BASE_RD}/{{idContato}}?token={{tokenRD ou RD_TOKEN}}
    Body baseado nos campos atuais do LEAD_STATE, podendo ser sobreposto pelos parâmetros.
    """
    cid = (idContato or LEAD_STATE.get("idContato") or "").strip()
    if not cid:
        return _safe_json({"ok": False, "erro": "idContato ausente"})
    token_use = (tokenRD or RD_TOKEN).strip()

    nome = (nomeLead or LEAD_STATE.get("nomeLead") or "").strip() or None
    email = (emailLead or LEAD_STATE.get("emailLead") or "").strip() or None
    numero = (numeroCliente or LEAD_STATE.get("numeroCliente") or "").strip()
    cargo = (cargoCliente or LEAD_STATE.get("cargoCliente") or "").strip() or None

    payload = {
        "name": nome,
        "title": cargo,
        "emails": [{"email": email}] if email else [],
        "phones": [{"type": "home", "phone": numero}] if numero else []
    }
    url = f"{BASE_RD}/{cid}?token={token_use}"
    try:
        resp = requests.put(url, headers=_rd_headers(), data=json.dumps(payload), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return _safe_json({"ok": True, "url": url, "payload": payload, "result": data})
    except (Timeout, HTTPError, RequestException) as e:
        return _safe_json({"ok": False, "erro": str(e), "url": url, "payload": payload})

# ------------------- Funções de Roteamento/Negócio ----------------------

@tool
def registrar_lead(status_lead: str) -> str:
    """
    Registra/atualiza o lead no RD:
      1) Se existir idContato → tentar rd_atualizar_contato
      2) Senão → rd_criar_contato
    Em seguida, seta LEAD_STATE['status_lead'] = status_lead.
    """
    status = (status_lead or "").strip().lower()
    if status not in {"quente", "frio", "desqualificado"}:
        return _safe_json({"ok": False, "erro": "status_lead inválido. Use 'quente', 'frio' ou 'desqualificado'."})

    # cria/atualiza
    if LEAD_STATE.get("idContato"):
        _ = rd_atualizar_contato.invoke({
            "idContato": LEAD_STATE["idContato"],
            "nomeLead": LEAD_STATE.get("nomeLead", ""),
            "emailLead": LEAD_STATE.get("emailLead", ""),
            "numeroCliente": LEAD_STATE.get("numeroCliente", ""),
            "cargoCliente": LEAD_STATE.get("cargoCliente", "")
        })
        result = json.loads(_)
    else:
        _ = rd_criar_contato.invoke({
            "nomeLead": LEAD_STATE.get("nomeLead", ""),
            "emailLead": LEAD_STATE.get("emailLead", ""),
            "numeroCliente": LEAD_STATE.get("numeroCliente", ""),
            "cargoCliente": LEAD_STATE.get("cargoCliente", "")
        })
        result = json.loads(_)

    LEAD_STATE["status_lead"] = status

    # Gera arquivo .txt com resultado do lead e explicação
    try:
        lead_info = {
            "Nome": LEAD_STATE.get("nomeLead"),
            "Email": LEAD_STATE.get("emailLead"),
            "Site": LEAD_STATE.get("siteEmpresa"),
            "Cargo": LEAD_STATE.get("cargoCliente"),
            "Número": LEAD_STATE.get("numeroCliente"),
            "Interesse": LEAD_STATE.get("interesse_produto"),
            "Tamanho do Time": LEAD_STATE.get("tamanho_time"),
            "Segmento": LEAD_STATE.get("segmento"),
            "Status do Lead": LEAD_STATE.get("status_lead"),
        }
        explanation = ""
        if status == "quente":
            explanation = (
                "Este lead é considerado QUENTE porque atende aos critérios:\n"
                "- Empresa com mais de 5 funcionários\n"
                "- Possui site ativo\n"
                "- Tem interesse real em algum produto/case da Robbu\n"
            )
        elif status == "frio":
            explanation = (
                "Este lead é considerado FRIO porque não atende a todos os critérios necessários:\n"
                "- Empresa com mais de 5 funcionários\n"
                "- Possui site ativo\n"
                "- Tem interesse real em algum produto/case da Robbu\n"
            )
        else:
            explanation = "Status do lead é DESQUALIFICADO ou outro."

        content_lines = ["Resultado da Qualificação do Lead\n", "===============================\n"]
        for k, v in lead_info.items():
            content_lines.append(f"{k}: {v or 'Não informado'}\n")
        content_lines.append("\nExplicação:\n")
        content_lines.append(explanation)

        filename = f"lead_result_{LEAD_STATE.get('emailLead', 'unknown').replace('@', 'at').replace('.', '')}{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(content_lines)
    except Exception as e:
        # Log error or ignore silently
        pass

    return _safe_json({"ok": True, "acao": "registrar_lead", "status_lead": status, "resultado_rd": result})

@tool
def enviar_para_comercial(lead_id: str = "", observacao: str = "") -> str:
    """
    Simula encaminhamento do lead quente para o time comercial.
    Idealmente, integre aqui um webhook/queue/CRM interno.
    """
    lid = (lead_id or LEAD_STATE.get("idContato") or "").strip()
    return _safe_json({"ok": True, "acao": "enviar_para_comercial", "lead_id": lid, "observacao": observacao or ""})

@tool
def enviar_para_suporte(assunto: str = "", descricao: str = "", cliente: bool = True) -> str:
    """
    Simula abertura de chamado/encaminhamento para suporte.
    """
    return _safe_json({"ok": True, "acao": "enviar_para_suporte", "assunto": assunto or "Suporte Robbu", "descricao": descricao or "", "cliente": bool(cliente)})

@tool
def coleta_leads() -> str:
    """
    Apenas sinaliza para o agente continuar a coleta orientada (uma pergunta por vez).
    """
    return _safe_json({"ok": True, "acao": "coleta_leads", "lead": LEAD_STATE})

@tool
def Get_finalizaCliente() -> str:
    """
    Finaliza o atendimento.
    """
    return _safe_json({"ok": True, "acao": "finalizar", "mensagem": "Atendimento finalizado. A Robbu agradece o contato!"})

# ETAPA 3: ORQUESTRAÇÃO (LANGGRAPH)


# Todas as ferramentas disponíveis para o modelo
tools = [
    pesquisa_tecnica_avancada_robbu,
    salvar_dado_lead,
    listar_dados_lead,
    avaliar_lead_quente,
    rd_listar_contatos,
    rd_criar_contato,
    rd_exibir_contato,
    rd_atualizar_contato,
    registrar_lead,
    enviar_para_comercial,
    enviar_para_suporte,
    coleta_leads,
    Get_finalizaCliente
]
tool_executor = ToolNode(tools)
model = llm.bind_tools(tools)

# PROMPT PRINCIPAL

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
Você é o assistente conversacional B2B da Robbu para qualificação de leads e suporte.
Atue conforme o roteiro abaixo e sempre chame as ferramentas quando necessário.

==================== 1 - Perfil ====================
- Objetivo: atuar como assistente conversacional B2B para qualificação de leads da Robbu. Identificar intenção, responder com precisão e direcionar para suporte ou comercial quando apropriado.

==================== 2 - Saudação ====================
- Mensagem inicial (apenas na primeira interação da conversa): “Olá, eu sou seu assistente virtual da Robbu. Como podemos te ajudar?”
- Aguarde a resposta do usuário antes de continuar.
- Estilo:
  - Linguagem: Português-BR (ou idioma do cliente).
  - Tom: Claro, empático, consultivo e confiante.
  - Destaque a inovação e expertise da Robbu.

==================== 3 - Detecção de Intenções ====================
Classifique a mensagem do usuário em uma das categorias abaixo:
Se a confiança < 0,6, classifique como Sem Intenção Definida e aplique sondagem.

1 - Carteiro Digital – Entrou por Engano
   Gatilhos: “boleto”, “2ª via”, “fatura”, “cobrança”, menções a marcas parceiras pedindo atendimento direto dessas marcas.
2 - Suporte
   Problemas técnicos, erros, dúvidas de uso, necessidade de atendimento para cliente existente.
3 - Informações (FAQ)
   Perguntas gerais sobre Robbu/produtos/capacidades/como funciona.
4 - Sem Intenção Definida
   Saudações, mensagens vagas.
5 - Interesse Comercial Direto
   Pedido explícito de proposta, demonstração ou reunião.

==================== 4 - Script de Conversa ====================
4.1 Intenção “Correio Digital – Entrou por Engano”
   Responder:
   "Para obter um novo boleto, entre em contato diretamente com a empresa emissora e solicite o reenvio, se necessário.
   A Robbu não é responsável pela geração do boleto, apenas pelo envio. Atuamos como um intermediário, conectando empresas e clientes de forma segura."
   Em seguida: “Posso te ajudar com algo mais?”
     - Se não: CHAME a função: Get_finalizaCliente.
     - Se sim: continue o fluxo conforme a nova intenção.

4.2 Intenção Suporte
   1. Pergunte ao usuário:  
   *"Você já é nosso cliente?"*
2. Se a resposta for *SIM*:  
  - chame a função: enviar_para_suporte
2.1 - Se for perguntas sobre suporte, envia a mensagem: "Nosso time de atendimento está disponível para esclarecer qualquer dúvida técnica." e pergunte posso te transferir para o nosso canal de suporte?
2.2 - após a resposta, se sim, execute a função: enviar_para_suporte
3. Se a resposta for *NÃO*:  
   3.1. Encaminhe o usuário imediatamente para o **Fluxo Qualificação (Seção 5).

4.3 Intenção Informações (FAQ)
   1) Responda usando a Base de conhecimento (abaixo). Se for técnico/erro/API → use pesquisa_tecnica_avancada_robbu.
   2) “Posso te ajudar com mais alguma dúvida?” Se não → Checkpoint Cliente.
   3) Checkpoint Cliente — “Você já é nosso cliente?”
      - Se SIM: CHAME enviar_para_suporte.
      - Se NÃO: abordagem consultiva breve (uma por vez):
           “Agora que você tirou suas dúvidas, me conta um pouco sobre sua empresa.”
           “Posso te apresentar uma solução que costuma gerar mais ganho para perfis como o seu?”
           “Você já conhece nossos produtos de IA?”
      → Se demonstrar interesse → Fluxo Qualificação.
      → Se não houver interesse → CHAME Get_finalizaCliente.

4.4 Sem Intenção Definida
   Sondagem leve (uma por vez):
     “No que posso ajudar hoje? Busca automação, atendimento inteligente ou campanhas?”
     “Você já conhece nossos chatbots com IA, o Invênio, o Carteiro Digital ou nossas campanhas de marketing?”
     “Conte rapidamente a necessidade da sua empresa para eu te direcionar melhor.”
   Ao detectar interesse:
     - Se SIM (cliente): CHAME enviar_para_suporte.
     - Se NÃO: Fluxo Qualificação.

4.5 Interesse Comercial Direto
   Vá direto para o Fluxo Qualificação.

==================== 5 - Fluxo Qualificação ====================
Sempre pergunte uma coisa por vez e registre as respostas com a ferramenta salvar_dado_lead.
5.1 Coleta base (pergunte uma por vez):
   - “Me conta um pouco sobre sua empresa (segmento e produto/serviço).”  → salvar_dado_lead(campo="segmento", valor=...)
   - “Quantas pessoas compõem o time hoje?” → salvar_dado_lead(campo="tamanho_time", valor=...)
   - “Vocês têm site para eu conhecer um pouco mais?” → quando a pessoa informar, chame salvar_dado_lead(campo="siteEmpresa", valor=...)
   - “Qual solução chamou mais sua atenção? (Chatbot IA, Invênio, Carteiro Digital, Campanhas, Outros)” → salvar_dado_lead(campo="interesse_produto", valor=...)

5.2 Preços
   Informe o valor apenas quando o cliente pergunte sobre perguntar sobre orçamento.
   Envie a mensagem padronizada:
- “Nossas propostas comerciais são personalizadas para cada tipo de negócio e necessidades, mas o plano inicial é de R$ 1.200/mês.
- Hoje esse valor está dentro do seu orçamento?
No valor de R$1.200 estão inclusos:
- licenças de atendimento ilimitadas
- licenças para automatização/chatbots ilimitadas
1.200 contatos por mês
- números de WhatsApp API ilimitados

Para eu ter uma ideia e ver se conseguimos negociar o valor, pode me informar qual o orçamento para investimento atual?”
- Se o cliente achar caro ou pedir desconto, pergunte se pode envia-lo para o time de comercial?
se sim, execute a função: enviar_para_comercial

5.3 Critérios para Lead Quente
   - Empresa com mais de 5 funcionários
   - Possui site ativo
   - Tem interesse real em algum produto/case da Robbu
   Quando houver dados suficientes, CHAME avaliar_lead_quente.

==================== 6 - Decisão ====================
- Se lead quente:
   “Ótimo! Podemos avançar para uma conversa com nosso time comercial para uma proposta personalizada?”
     - Se SIM:
        1) Garanta que temos nome, e-mail, cargo e (opcional) número de telefone:
           → salvar_dado_lead para cada um que chegar em resposta:
              - salvar_dado_lead(campo="nomeLead", valor=...)
              - salvar_dado_lead(campo="emailLead", valor=...)
              - salvar_dado_lead(campo="cargoCliente", valor=...)
              - salvar_dado_lead(campo="numeroCliente", valor=...)
        2) CHAME registrar_lead(status_lead="quente")
        3) CHAME enviar_para_comercial(lead_id=LEAD_STATE['idContato'])
     - Se NÃO: CHAME coleta_leads (para continuar a coleta mínima e encerrar cordialmente).

- Se lead não apto:
   CHAME registrar_lead(status_lead="frio") → Fluxo Contato (Seção 7).

==================== 7 - Fluxo Contato ====================
- Se Quente: informe transferência para comercial e CHAME enviar_para_comercial.
- Se Frio: agradeça e informe que nossa equipe comercial analisará e retornará assim que possível.

==================== 8 - Fluxo Novo Contato ====================
- Se Quente: transfira para comercial (enviar_para_comercial).
- Se Frio: agradeça e informe retorno posterior.

==================== 9 - Regras Gerais ====================
- Uma pergunta por vez; sempre aguarde resposta.
- A base de conhecimento da Robbu é a sua principal referência.
Sempre que receber perguntas sobre a Robbu, seus produtos, serviços ou cases de sucesso, utilize exclusivamente as informações contidas nessa base para formular suas respostas.
- Não invente informações fora da base de conhecimento.
- Não faça contas
- Não deve falar sobre a CrewAI, nem sobre o que é um agente, nem sobre como funciona a CrewAI.
- Não deve falar sobre o que é um LLM, nem sobre como funciona o modelo de linguagem.
- Não deve falar sobre o que é um assistente virtual, nem sobre como funciona um assistente virtual.
- Nunca encerrar o diálogo com respostas fechadas.
- Evite jargão técnico sem necessidade.
- Não prometa ações fora do chat sem acionar a função correspondente.
- Ao captar: email, nome, site, cargo → CHAME salvar_dado_lead com os campos: "emailLead", "nomeLead", "siteEmpresa", "cargoCliente".
- Para consultar/criar/atualizar contato no RD, use:
   - rd_listar_contatos(emailLead=?)
   - rd_criar_contato(...)
   - rd_exibir_contato(idContato=?)
   - rd_atualizar_contato(idContato=?, ...)

==================== 10 - Base de conhecimento ====================
{ROBBU_KNOWLEDGE_BASE} 

11 - IMPORTANTE:
- Para informações de contato: nunca invente dados; peça ao usuário e salve com salvar_dado_lead.
- Seja claro, empático, consultivo e confiante.
- Não prometa ações fora do chat sem acionar a função correspondente.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Combina o prompt com o modelo para criar a cadeia principal do agente
chain = prompt | model

# --- Nós do Grafo ---
def call_model(state):
    messages = state["messages"]
    try:
        response = chain.invoke({"messages": messages})
        return {"messages": [response]}
    except Exception as e:
        
        try:
            from langchain_core.messages import AIMessage
            openrouter_result = call_openrouter_model(state)
            openrouter_result["messages"].append(
                AIMessage(content=f"[INFO] Fallback to OpenRouter due to error: {str(e)}")
            )
            return openrouter_result
        except Exception as e2:
            from langchain_core.messages import AIMessage
            return {"messages": messages + [AIMessage(content=f"[ERRO] Falha no OpenAI e OpenRouter: {str(e)} | {str(e2)}")]}

def call_openrouter_model(state): #quando a open ai falhar vai chamar o segundo modelo, modelo gratuito teste.
    

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("Chave da API do OpenRouter não encontrada. Verifique seu arquivo .env")

    messages = state["messages"]
    user_message = None
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
            user_message = msg
            break
    if not user_message:
        user_message = messages[-1]

    prompt = getattr(user_message, "content", None) or str(user_message)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"[ERRO_OPENROUTER:{str(e)}]"

    # status
    from langchain_core.messages import AIMessage
    return {"messages": state["messages"] + [AIMessage(content=reply)]}

def should_continue(state):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_executor)
workflow.add_node("openrouter", call_openrouter_model)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "openrouter": "openrouter", "end": END},
)
workflow.add_edge("action", "agent")
workflow.add_edge("openrouter", END)
agent_graph_leads = workflow.compile()