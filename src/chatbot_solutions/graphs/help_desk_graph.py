import os
import operator
from typing import Annotated, TypedDict, List, Dict, Any
import json
from datetime import datetime
import re
import ast

# --- Dependências Essenciais ---
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from schemas.usuario_schema import AgentState

# --- Dependências da CrewAI ---
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError


# ETAPA 1: CONFIGURAÇÕES E CONHECIMENTO BASE

# Carrega variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Chave da API da OpenAI não encontrada. Verifique seu arquivo .env")

# Define o LLM principal usado por todos os agentes da CrewAI e o agente principal
llm = ChatOpenAI(
    model="gpt-4.1", 
    api_key=API_KEY,
    temperature=0.8
)


# Este é o conhecimento base "embutido" no agente.
ROBBU_KNOWLEDGE_BASE = """
Nossa história: Fundada em 2016, a Robbu é líder em soluções de automação de comunicação. Nosso produto permite que clientes organizem o relacionamento com seus consumidores em uma solução omnichannel, totalmente personalizável e integrada a diversos sistemas.
Sinergia e tecnologia: Acreditamos que o sucesso está na comunicação personalizada. Combinamos nosso DNA de customer experience com o melhor da tecnologia para transformar o contato entre marcas e clientes.
Parcerias e Alcance: Somos parceiros do Google, Meta e Microsoft, provedores oficiais do WhatsApp Business API e fomos Empresa Destaque do Facebook (Meta) em 2021. Temos mais de 800 clientes em 26 países.
Principais produtos: Plataforma omnichannel (WhatsApp, Instagram, etc.), Chatbots com IA, Automação de marketing, Relatórios, Integrações com CRMs/ERPs, e uma API robusta.
Segurança: Seguimos rigorosamente as normas da LGPD.

***Se o ususário perguntar sobre as lideranças da Robbu, responda:***
CEO: Álvaro Garcia Neto
CO-Founder: Helber Campregher
"""


# Essas informações são usadas pela CrewAI quando ela é acionada.
ROBBU_DOCS_CONTEXT = [
    {"name": "Templates WhatsApp", "url": "https://docs.robbu.global/docs/center/como-criar-templates-whatsapp"},
    {"name": "Edição de Tags (Live )", "url": "https://docs.robbu.global/docs/live/edicao-de-tags"},
    {"name": "Configurações Gerais da Conta", "url": "https://robbu.mintlify.app/docs/center/configuracoes-gerais-da%20conta"},
    {"name": "Gerenciar Host de Acesso", "url": "https://docs.robbu.global/docs/center/gerenciar-host-de-acesso#o-que-e-o-gerenciar-hosts-de-acesso"},
    {"name": "Como Criar Agendamento (Live )", "url": "https://docs.robbu.global/docs/live/como-criar-agendamento"},
    {"name": "Carteiro Digital API", "url": "https://docs.robbu.global/docs/carteiro-digital/carteiro-digital-api"},
    {"name": "Carteiro Digital", "url": "https://docs.robbu.global/docs/carteiro-digital/carteiro-digital"},
    {"name": "Gestão de Frases Prontas", "url": "https://docs.robbu.global/docs/center/gestao-frases-prontas"},
    {"name": "Restrições", "url": "https://docs.robbu.global/docs/center/restricoes"},
    {"name": "Campanhas: Público e Importação", "url": "https://docs.robbu.global/docs/center/campanhas-publico-importacao"},
    {"name": "Criando Campanha SMS", "url": "https://docs.robbu.global/docs/center/criando-campanha-sms"},
    {"name": "Bibliotecas de Mídias", "url": "https://docs.robbu.global/docs/center/bibliotecas-de-midias"},
    {"name": "Criar Campanhas de WhatsApp", "url": "https://docs.robbu.global/docs/center/campanhas-de-whatsapp"},
    {"name": "Canais de Atendimento", "url": "https://docs.robbu.global/docs/center/canais-atendimento"},
    {"name": "Canal WhatsApp", "url": "https://docs.robbu.global/docs/center/canal-whatsapp"},
    {"name": "Como Alterar Imagem da Linha WhatsApp", "url": "https://robbu.mintlify.app/docs/center/como-alterar-imagem-da-linha-whatsapp"},
    {"name": "Criação de Contatos Invenio Center", "url": "https://robbu.mintlify.app/docs/center/criacao-de-contatos-invenio-center"},
    {"name": "Exportação de Conversas (Live )", "url": "https://robbu.mintlify.app/docs/live/exportacao-de-conversas"},
    {"name": "Fila de Atendimento", "url": "https://robbu.mintlify.app/docs/center/fila-de-atendimento"},
    {"name": "Filtros de Busca de Contatos", "url": "https://robbu.mintlify.app/docs/center/filtros-de-busca-de-contatos"},
    {"name": "Lista de Desejos (Live )", "url": "https://robbu.mintlify.app/docs/live/lista-de-desejos"},
    {"name": "Métodos de Distribuição (Live )", "url": "https://docs.robbu.global/docs/live/metodos-de-distribuicao"},
    {"name": "Sessão de 24 Horas no WhatsApp", "url": "https://robbu.mintlify.app/docs/live/sessao-de-24horas-no-whatsapp"},
    {"name": "Usuários", "url": "https://docs.robbu.global/docs/center/usuarios"},
    {"name": "Relatórios", "url": "https://docs.robbu.global/docs/center/relatorios"},
    {"name": "Webchat", "url": "https://docs.robbu.global/docs/center/web-chat"},
    {"name": "KPI Eventos", "url": "https://docs.robbu.global/docs/center/dashboard-kpi-eventos"},
    {"name": "Privacidade e LGPD", "url": "https://docs.robbu.global/docs/center/privacidade-e-protecao"}
]

META_DOCS_CONTEXT = [
    {"name": "Códigos de Erro da API", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/support/error-codes/?locale=pt_BR"},
    {"name": "Migração de On-Premises para a Nuvem", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/guides/migrating-from-onprem-to-cloud?locale=pt_BR"},
    {"name": "Contas Comerciais do WhatsApp", "url": "https://developers.facebook.com/docs/whatsapp/overview/business-accounts"},
    {"name": "Números de Telefone (Cloud API)", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/phone-numbers"},
    {"name": "Configuração de Webhooks", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks"},
    {"name": "Documentação da API do WhatsApp", "url": "https://developers.facebook.com/docs/whatsapp/cloud-api/"},
    {"name": "Política de Privacidade do WhatsApp", "url": "https://www.whatsapp.com/legal/privacy-policy?lang=pt_BR"},
    {"name": "Política de Uso da API do WhatsApp", "url": "https://www.whatsapp.com/legal/business-policy?lang=pt_BR"}
    ]


# ETAPA 2: FERRAMENTAS - CREW COMO FERRAMENTA

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
        # CORREÇÃO: Introduzido um agente 'analisador' para escolher a URL semanticamente.
        analisador = Agent(
            role="Analisador de Documentos",
            goal="Analisar a pergunta de um usuário e mapeá-la para a URL mais relevante de uma lista de documentos.",
            backstory="Você é um especialista em entender a intenção por trás de uma pergunta e encontrar o documento exato que a responde em uma lista pré-definida.",
            llm=llm,
            verbose=True
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

        # --- ETAPA 1: ANÁLISE E ESCOLHA DA URL ---
        documentos_formatados = "\n".join([f"- Título do Documento: '{doc['name']}', URL: {doc['url']}" for doc in ROBBU_DOCS_CONTEXT + META_DOCS_CONTEXT])
        
        tarefa_analise = Task(
            description=f"""
            Analise a pergunta do usuário: "{query}".
            Sua tarefa é encontrar a URL do documento mais relevante na lista de 'Documentos Conhecidos' para responder a essa pergunta.
            Concentre-se no significado da pergunta e no 'Título do Documento'.

            Documentos Conhecidos:
            {documentos_formatados}
            """,
            expected_output="A URL exata do documento mais relevante. Se nenhum for relevante, retorne 'N/A'.",
            agent=analisador
        )
        
        crew_analise = Crew(agents=[analisador], tasks=[tarefa_analise], process=Process.sequential, verbose=True, telemetry=False)
        url = crew_analise.kickoff()

        # --- VALIDAÇÃO DA URL E HANDOFF HUMANO ---
        if "N/A" in url or not str(url).startswith("http"):
            return "Não localizei uma página específica para essa dúvida em nossa documentação. Você gostaria de ser transferido para um atendente?"

        # --- ETAPAS 2 E 3: EXTRAÇÃO E REDAÇÃO ---
        tarefa_extracao = Task(
            description=f"Extraia o conteúdo da URL: {url}",
            expected_output="O texto limpo da página.",
            agent=extrator
        )
        
        tarefa_redacao = Task(
            description=f"""
            Produza uma resposta técnica objetiva para a pergunta '{query}', usando o conteúdo extraído.
            Sua resposta deve ser em português, profissional, competente e concisa.

            IMPORTANTE: Se o conteúdo extraído contiver a string '[ERRO_EXTRACAO', isso significa que a extração da página falhou.
            Nesse caso, sua resposta final DEVE ser:
            'Ouve um erro ao tentar processar a documentação. Você pode continuar acessando o conteúdo diretamente através deste link: {url}'
            """,
            expected_output="Resposta técnica profissional ou a mensagem de erro de extração.",
            agent=redator,
            context=[tarefa_extracao]
        )
        
        crew_processamento = Crew(
            agents=[extrator, redator],
            tasks=[tarefa_extracao, tarefa_redacao],
            process=Process.sequential,
            verbose=True,
            telemetry=False
        )
        
        resultado_final = crew_processamento.kickoff()
        
        return resultado_final if resultado_final else f"Não foi possível processar a solicitação para a URL: {url}"

@tool
def pesquisa_tecnica_avancada_robbu(query: str) -> str:
    """
    Use esta ferramenta para responder a perguntas técnicas específicas sobre a plataforma Robbu, funcionalidades como 'templates' e 'relatórios', ou sobre a API do WhatsApp, como 'códigos de erro' e 'contas comerciais'. A ferramenta busca na documentação oficial e retorna uma resposta técnica completa.
    """
    executor = TechnicalCrewExecutor()
    return executor.run(query)


# ETAPA 3: ORQUESTRAÇÃO (LANGGRAPH COM AGENTE ÚNICO FUNCIONANDO DE FORMA GERAL)


# Define o estado do nosso agente
# Define as ferramentas que o agente pode usar
tools = [pesquisa_tecnica_avancada_robbu]
tool_executor = ToolNode(tools)

# Associa as ferramentas ao LLM. Isso permite que o LLM "veja" as ferramentas.
model = llm.bind_tools(tools)

# PROMPT DO AGENTE PRINCIPAL
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é o agente help desk especialista da Robbu. Você é um agente profissional, treinado para responder perguntas técnicas sobre a plataforma Robbu e a API do WhatsApp da Meta.

<Apresentação>
- Na primeira interação, você se apresenta como o agente help desk da Robbu e pergunta como pode ajudar o usuário.
</Apresentação>

<Persona e Tom de Voz>
- Você é profissional, prestativo e direto.
- Comunique-se de forma clara e objetiva. Use a primeira pessoa do plural ('nossos') ao falar sobre a Robbu e trata o interlocutor como 'você'.
- Use gírias moderadas (exemplo: "legal", "bacana", "maravilha", "super") e emojis.
- Evitar jargões técnicos desnecessários e linguagem excessivamente rebuscada. Usar uma comunicação clara, amigável e acessível, como um profissional atencioso que fala a linguagem do cliente.
- Deve ser objetivo, mas fornecer explicações suficientes quando as dúvidas forem mais complexas. Respostas curtas, mas informativas.
</Persona e Tom de Voz>

<Perfil dos Clientes>
- O usuário é um cliente da Robbu que busca ajuda técnica ou informações sobre a plataforma.
- Usuários que precisam esclarecer dúvidas sobre a plataforma Robbu, funcionalidades como 'templates' e 'relatórios', ou sobre a API do WhatsApp, como 'códigos de erro' e 'contas comerciais'.
- Os usuários podem ter diferentes níveis de conhecimento técnico, desde iniciantes até desenvolvedores experientes.
</Perfil dos Clientes>

<Restrições Gerais>
- Você não responde perguntas pessoais, políticas ou que não estejam relacionadas à Robbu, não responde sobre outros produtos ou serviços, e não especula sobre assuntos desconhecidos.
- Se o assunto estiver completamente fora do escopo da Robbu, informe educadamente que não pode ajudar com aquele tópico.
- Você só deve falar sobre a Robbu e suas funcionalidades, não deve falar sobre produtos ou serviços de terceiros como blip, mundiale e etc.
- Não realiza calculos financeiros, nem calculos basicos como 2+2, nem perguntas de lógica ou enigmas.
- Não deve falar sobre a CrewAI, nem sobre o que é um agente, nem sobre como funciona a CrewAI.
- Não deve falar sobre o que é um LLM, nem sobre como funciona o modelo de linguagem.
- Não deve falar sobre o que é um assistente virtual, nem sobre como funciona um assistente virtual.
</Restrições Gerais>

**Seu Conhecimento Base sobre a Robbu (Responda diretamente se a resposta da sua pergunta estiver aqui):**
{robbu_knowledge}

**Suas Ferramentas:**
Você tem acesso a ferramentas para buscar informações que não estão no seu conhecimento base.
Você **DEVE** decidir, com base na pergunta do usuário, se pode responder diretamente ou se precisa usar uma ferramenta.
- Para perguntas técnicas sobre funcionalidades específicas (templates, relatórios, integrações), APIs, ou erros, **USE** a ferramenta `pesquisa_tecnica_avancada_robbu`.
- Se o usuário pedir explicitamente para falar com uma pessoa, ou se a busca falhar e a opção for oferecida, **USE** a ferramenta `transferir_atendente_humano`.
- Para saudações, agradecimentos, ou perguntas institucionais simples (cobertas no seu conhecimento base), **NÃO USE** ferramentas, responda diretamente usando seu conhecimento base.

**Fluxo da Conversa:**

1. Analise a pergunta do usuário.
2. Se a resposta estiver no seu "Conhecimento Base", responda diretamente.
3. Se for uma pergunta técnica complexa sobre a plataforma Robbu, configurações ou a API do WhatsApp, chame a ferramenta `pesquisa_tecnica_avancada_robbu` e use o resultado para formular sua resposta final.
4. A resposta final deve ser explicativa e clara, abordando os pontos principais para que o usuário entenda a solução ou informação fornecida. Se possivel no formato de passo a passo.
""".format(robbu_knowledge=ROBBU_KNOWLEDGE_BASE),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Combina o prompt com o modelo para criar a cadeia principal do agente
chain = prompt | model

# --- Lógica do Grafo ---

# Nó 1: Chama o modelo (LLM) para decidir o que fazer
def call_model(state):
    messages = state["messages"]
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}

# Nó 2: Decide se deve continuar (usar uma ferramenta) ou terminar
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# Construção do Grafo
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
app = workflow.compile()
print("✅ Agente Inteligente e memória compilado com sucesso!")


# ETAPA 4: INTERFACE DE TESTE (COM MEMÓRIA E TRADUÇÃO)

def processar_mensagem(mensagem: str, history: List[BaseMessage]) -> (str, List[BaseMessage]):
    """Processa uma nova mensagem do usuário, considerando o histórico da conversa."""
    try:
        inputs = {"messages": history + [HumanMessage(content=mensagem)]}
        final_state = app.invoke(inputs, {"recursion_limit": 10})
        assistant_response = final_state['messages'][-1].content
        return assistant_response, final_state['messages']
    except Exception as e:
        print(f"Erro ao processar mensagem: {e}")
        return "Ocorreu um erro ao processar sua solicitação.", history

def detect_language_with_llm(text: str) -> str:
    """Detecta o idioma de um texto usando o LLM."""
    prompt = f"""
    Qual é o idioma do texto a seguir? Responda APENAS com o código de idioma ISO 639-1 de duas letras (ex: 'en', 'es', 'pt').

    Texto: "{text}"
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip().lower()
    except Exception:
        return 'pt'

def translate_with_llm(text: str, target_language: str) -> str:
    """Traduz um texto para o idioma de destino usando o LLM."""
    prompt = f"""
    Traduza o seguinte texto para o idioma com o código '{target_language}'.
    Mantenha o tom, a formatação e os emojis o máximo possível.

    Texto para traduzir:
    "{text}"
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return text

def executar_interface_teste():
    print("\n" + "="*60)
    print("HELP DESK ROBBU")
    print("="*60)
    print("Digite 'sair' para encerrar.")
    print("-"*60)
    
    conversation_history: List[BaseMessage] = []

    while True:
        try:
            mensagem = input("\nVocê: ").strip()
            if mensagem.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando.")
                break
            if not mensagem:
                continue
            
            print("Assistente: ...pensando...")

            detected_lang = detect_language_with_llm(mensagem)
            
            resposta_pt, conversation_history = processar_mensagem(mensagem, conversation_history)
            
            if detected_lang != 'pt':
                print(f"Assistente: ...traduzindo para '{detected_lang}'...")
                resposta_final = translate_with_llm(resposta_pt, detected_lang)
            else:
                resposta_final = resposta_pt

            print("Assistente:", resposta_final)
        except KeyboardInterrupt:
            print("\nEncerrado.")
            break
        except Exception as e:
            print(f"Erro na interface: {e}")
