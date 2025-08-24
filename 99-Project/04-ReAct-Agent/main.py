# -*- coding: utf-8 -*-
# Streamlit 및 기본 라이브러리
import streamlit as st
import os
import uuid
from typing import List

# LangChain 관련 라이브러리
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_graph

# 환경 설정
from dotenv import load_dotenv

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
logging.langsmith("LangChain-Tutorial")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# Streamlit 앱 제목 설정
st.title("ReAct Agent 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
if "tools" not in st.session_state:
    st.session_state["tools"] = []
if "memory" not in st.session_state:
    # 전역 메모리 인스턴스 - 한 번만 생성되고 계속 유지됨
    st.session_state["memory"] = MemorySaver()
if "current_tool_config" not in st.session_state:
    # 도구 설정 변경 감지를 위한 상태
    st.session_state["current_tool_config"] = None


def create_web_search_tool() -> TavilySearch:
    """웹 검색 도구 생성"""
    web_search = TavilySearch(
        topic="general",
        max_results=3,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        format_output=False,
    )
    web_search.name = "web_search"
    web_search.description = (
        "Use this tool to search on the web for current information, news, and general topics. "
        "Perfect for finding real-time data, latest news, or any information not in your training data."
    )
    return web_search


def create_python_repl_tool() -> PythonREPLTool:
    """Python REPL 코드 실행 도구 생성"""
    python_tool = PythonREPLTool()
    python_tool.name = "python_repl"
    python_tool.description = (
        "A Python shell that can execute Python code and return results. "
        "Use this for calculations, data analysis, generating charts, and any computational tasks. "
        "Remember to use print() to see output results."
    )
    return python_tool


def create_pdf_retriever_tool(uploaded_file) -> object:
    """PDF 리트리버 도구 생성"""
    if uploaded_file is None:
        return None
    
    # PDF 파일 저장
    file_content = uploaded_file.read()
    file_path = f"./.cache/files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # PDF 문서 로드 및 분할
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    
    # 벡터 저장소 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    # 리트리버 도구 생성
    retriever_tool = create_retriever_tool(
        retriever,
        "pdf_retriever",
        f"Search and return information from the uploaded PDF file: {uploaded_file.name}. "
        f"This tool contains the full content of the document and can answer questions about it.",
        document_prompt=PromptTemplate.from_template(
            "<document><content>{page_content}</content><metadata><source>{source}</source><page>{page}</page></metadata></document>"
        ),
    )
    return retriever_tool


def create_react_agent_executor(selected_tools: List, model_name: str = "gpt-4.1", temperature: float = 0.1):
    """ReAct Agent 생성"""
    # LLM 모델 설정
    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    # 기존 메모리 인스턴스 사용 (대화 기록 유지)
    memory = st.session_state["memory"]
    
    # ReAct Agent 생성
    agent_executor = create_react_agent(
        model, 
        selected_tools, 
        checkpointer=memory
    )
    
    return agent_executor


def print_messages():
    """저장된 대화 기록을 화면에 표시"""
    if st.session_state["messages"]:
        for msg_data in st.session_state["messages"]:
            # 메시지가 딕셔너리 형태인 경우 (도구 호출 정보 포함)
            if isinstance(msg_data, dict):
                role = msg_data.get("role")
                content = msg_data.get("content")
                tool_calls = msg_data.get("tool_calls", [])
                
                with st.chat_message(role):
                    # 도구 호출 정보가 있는 경우 먼저 표시
                    if tool_calls:
                        with st.expander(f"🛠️ 도구 호출 정보", expanded=False):
                            for i, tool_call in enumerate(tool_calls, 1):
                                st.markdown(f"**{i}. {tool_call['name']}**")
                                
                                # 도구 호출 인자 표시
                                if tool_call['args']:
                                    st.markdown("📝 **호출 인자:**")
                                    for key, value in tool_call['args'].items():
                                        # 값이 너무 긴 경우 축약
                                        if isinstance(value, str) and len(value) > 100:
                                            value = value[:100] + "..."
                                        st.markdown(f"  • `{key}`: {value}")
                                
                                # 도구 실행 결과 표시
                                if 'result' in tool_call:
                                    st.markdown("📊 **실행 결과:**")
                                    st.code(tool_call['result'], language="text")
                                
                                if i < len(tool_calls):
                                    st.divider()
                    
                    # AI 응답 표시
                    st.markdown(content)
            
            # 기존 ChatMessage 형태인 경우 (하위 호환성)
            else:
                st.chat_message(msg_data.role).write(msg_data.content)
    else:
        st.info("💭 안녕하세요! ReAct Agent와 대화해보세요. 도구를 사용하여 다양한 작업을 수행할 수 있습니다.")


def add_message(role: str, message: str, tool_calls: list = None):
    """새로운 대화 메시지를 세션 상태에 저장 (도구 호출 정보 포함)"""
    msg_data = {
        "role": role,
        "content": message,
        "tool_calls": tool_calls or []
    }
    st.session_state["messages"].append(msg_data)


# 사이드바 UI 구성
with st.sidebar:
    st.header("⚙️ Agent 설정")
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())  # 새로운 thread_id 생성
        st.rerun()
    
    st.divider()
    
    # 모델 설정
    st.subheader("✅ 모델 설정")
    selected_model = st.selectbox(
        "LLM 모델 선택",
        ["gpt-4.1", "gpt-4.1-mini"],
        index=0,
        help="사용할 언어모델을 선택하세요."
    )
    
    temperature = st.slider(
        "🌡️ Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="0에 가까울수록 정확하고 일관된 답변, 1에 가까울수록 창의적인 답변"
    )
    
    # 답변 길이 조절 슬라이더
    response_length = st.slider(
        "📏 답변 길이 설정",
        min_value=1,
        max_value=5,
        value=3,
        help="1: 간단 (1-2문장), 2: 짧음 (1문단), 3: 보통 (2-3문단), 4: 자세함 (4-5문단), 5: 매우 자세함 (5문단 이상)"
    )
    
    st.divider()
    
    # 도구 선택
    st.subheader("🛠️ 도구 선택")
    
    # 웹 검색 도구
    use_web_search = st.checkbox(
        "🌐 웹 검색 도구",
        value=True,
        help="실시간 웹 검색을 통해 최신 정보를 찾습니다."
    )
    
    # Python 코드 실행 도구
    use_python_repl = st.checkbox(
        "🐍 Python 코드 실행 도구",
        value=True,
        help="Python 코드를 실행하여 계산, 데이터 분석, 차트 생성 등을 수행합니다."
    )
    
    # PDF 업로드 및 검색 도구
    st.subheader("📄 PDF 문서 도구")
    uploaded_pdf = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        help="PDF 파일을 업로드하면 문서 내용을 검색할 수 있는 도구가 추가됩니다."
    )
    
    use_pdf_retriever = uploaded_pdf is not None
    


# 도구 설정 및 Agent 생성
def setup_agent():
    """선택된 도구들을 기반으로 Agent 설정"""
    # 현재 도구 설정을 문자열로 생성 (변경 감지용)
    pdf_name = uploaded_pdf.name if uploaded_pdf else None
    current_config = {
        'web_search': use_web_search,
        'python_repl': use_python_repl, 
        'pdf_retriever': use_pdf_retriever,
        'pdf_name': pdf_name,
        'model': selected_model,
        'temperature': temperature,
        'response_length': response_length
    }
    config_str = str(sorted(current_config.items()))
    
    # 설정이 변경된 경우에만 Agent 재생성
    if (st.session_state["current_tool_config"] != config_str or 
        st.session_state["agent"] is None):
        
        tools = []
        
        # 웹 검색 도구 추가
        if use_web_search:
            tools.append(create_web_search_tool())
        
        # Python REPL 도구 추가
        if use_python_repl:
            tools.append(create_python_repl_tool())
        
        # PDF 리트리버 도구 추가
        if use_pdf_retriever:
            pdf_tool = create_pdf_retriever_tool(uploaded_pdf)
            if pdf_tool:
                tools.append(pdf_tool)
        
        # Agent 생성 (도구가 있을 때만)
        if tools:
            agent = create_react_agent_executor(
                selected_tools=tools,
                model_name=selected_model,
                temperature=temperature
            )
            # 설정 업데이트
            st.session_state["current_tool_config"] = config_str
            return agent, tools
        else:
            st.session_state["current_tool_config"] = config_str
            return None, []
    
    # 설정이 변경되지 않았으면 기존 Agent 사용
    else:
        return st.session_state["agent"], st.session_state["tools"]


# Agent 설정
agent, current_tools = setup_agent()
st.session_state["agent"] = agent
st.session_state["tools"] = current_tools

# 메인 채팅 인터페이스
print_messages()

# 사용자 입력
user_input = st.chat_input("💬 무엇이든 물어보세요! Agent가 필요한 도구를 사용하여 답변드립니다.")

# 사용자 질문 처리
if user_input:
    if st.session_state["agent"] is None:
        st.error("⚠️ 먼저 사이드바에서 사용할 도구를 선택해주세요.")
    else:
        # 사용자 질문 표시
        st.chat_message("user").write(user_input)
        add_message("user", user_input)
        
        # Agent 응답 생성 및 표시
        with st.chat_message("assistant"):
            
            try:
                # 설정 정보
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                inputs = {"messages": [("human", user_input)]}
                
                # Agent 실행 (스트리밍)
                full_response = ""
                tool_calls = []
                
                with st.spinner("🤔 Agent가 생각하고 있습니다..."):
                    # Agent 실행하여 응답 생성
                    response = st.session_state["agent"].invoke(inputs, config)
                    
                    # 모든 메시지를 분석하여 도구 호출 및 AI 응답 추출
                    if response and "messages" in response:
                        for msg in response["messages"]:
                            # 도구 호출 메시지 확인
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_info = {
                                        'name': tool_call.get('name', 'Unknown Tool'),
                                        'args': tool_call.get('args', {}),
                                        'id': tool_call.get('id', 'unknown')
                                    }
                                    tool_calls.append(tool_info)
                            
                            # 도구 실행 결과 메시지 확인
                            elif hasattr(msg, 'type') and msg.type == 'tool':
                                # 기존 도구 호출 정보에 결과 추가
                                tool_id = getattr(msg, 'tool_call_id', None)
                                content = getattr(msg, 'content', '')
                                for tool_call in tool_calls:
                                    if tool_call['id'] == tool_id:
                                        tool_call['result'] = content[:200] + "..." if len(content) > 200 else content
                                        break
                        
                        # AI의 최종 응답 추출
                        ai_messages = [msg for msg in response["messages"] if hasattr(msg, 'type') and msg.type == 'ai']
                        if ai_messages:
                            full_response = ai_messages[-1].content
                        else:
                            # AIMessage 타입이 아닌 경우 content 속성 직접 접근
                            for msg in reversed(response["messages"]):
                                if hasattr(msg, 'content') and msg.content.strip():
                                    full_response = msg.content
                                    break
                
                # 도구 호출 정보 표시 (확장 가능한 섹션으로)
                if tool_calls:
                    with st.expander(f"🛠️ 도구 호출 정보", expanded=False):
                        for i, tool_call in enumerate(tool_calls, 1):
                            st.markdown(f"**{i}. {tool_call['name']}**")
                            
                            # 도구 호출 인자 표시
                            if tool_call['args']:
                                st.markdown("📝 **호출 인자:**")
                                for key, value in tool_call['args'].items():
                                    # 값이 너무 긴 경우 축약
                                    if isinstance(value, str) and len(value) > 100:
                                        value = value[:100] + "..."
                                    st.markdown(f"  • `{key}`: {value}")
                            
                            # 도구 실행 결과 표시
                            if 'result' in tool_call:
                                st.markdown("📊 **실행 결과:**")
                                st.code(tool_call['result'], language="text")
                            
                            if i < len(tool_calls):
                                st.divider()
                
                # AI 최종 응답 표시
                if full_response:
                    st.markdown(full_response)
                    add_message("assistant", full_response, tool_calls)
                else:
                    st.error("죄송합니다. 응답을 생성하는 중 문제가 발생했습니다.")
                    
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")
                st.info("💡 모델 설정을 확인하거나 다시 시도해보세요.")

# 사이드바 하단에 현재 설정 정보 표시
with st.sidebar:
    st.divider()
    st.markdown("### 📊 현재 설정")
    st.caption(f"**모델:** {selected_model}")
    st.caption(f"**Temperature:** {temperature}")
    st.caption(f"**답변 길이:** {response_length}")
    st.caption(f"**Thread ID:** {st.session_state['thread_id'][:8]}...")
    st.caption(f"**활성 도구 개수:** {len(current_tools)}")
    
    # 도구별 상세 정보
    if current_tools:
        with st.expander("🔧 도구 상세 정보"):
            for i, tool in enumerate(current_tools):
                tool_name = getattr(tool, 'name', f'Tool {i+1}')
                tool_desc = getattr(tool, 'description', 'No description available')[:100] + "..."
                st.caption(f"**{tool_name}:** {tool_desc}")