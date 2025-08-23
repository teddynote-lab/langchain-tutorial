# Streamlit 및 기본 라이브러리
import streamlit as st
import os

# LangChain 관련 라이브러리
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging

# 환경 설정
from dotenv import load_dotenv

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
logging.langsmith("LangChain-Tutorial")

# 캐시 디렉토리 생성 (파일 업로드 및 임베딩 저장을 위함)
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 벡터 임베딩 저장 폴더
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# Streamlit 앱 제목 설정
st.title("📄 PDF 기반 QA 시스템")

# Streamlit 세션 상태 초기화 (앱 재실행 시에도 대화 기록 유지)
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 리스트 초기화
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # RAG 체인 초기화 (파일 업로드 전까지는 None)
    st.session_state["chain"] = None

# 사이드바 UI 구성
with st.sidebar:
    # 대화 기록 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화")

    # PDF 파일 업로드 위젯
    uploaded_file = st.file_uploader(
        "📎 PDF 파일 업로드",
        type=["pdf"],
        help="PDF 파일을 업로드하면 문서 내용을 기반으로 질문에 답변합니다.",
    )

    # LLM 모델 선택 드롭다운
    selected_model = st.selectbox(
        "LLM 모델 선택",
        ["gpt-4.1", "gpt-4.1-mini"],
        index=0,
        help="사용할 언어모델을 선택하세요.",
    )

    # 답변 길이 조절 슬라이더
    response_length = st.slider(
        "📏 답변 길이 설정",
        min_value=1,
        max_value=5,
        value=3,
        help="1: 간단 (1-2문장), 2: 짧음 (1문단), 3: 보통 (2-3문단), 4: 자세함 (4-5문단), 5: 매우 자세함 (5문단 이상)",
    )

    # 검색할 문서 개수 조절 슬라이더
    search_k = st.slider(
        "🔍 검색 문서 개수 설정",
        min_value=4,
        max_value=10,
        value=6,
        help="질문과 관련된 문서 청크를 몇 개까지 검색할지 설정합니다. 많을수록 더 많은 정보를 참고하지만 처리 시간이 길어집니다.",
    )


# 이전 대화 기록을 화면에 출력하는 함수
def print_messages():
    """저장된 대화 기록을 순서대로 화면에 표시"""
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 세션 상태에 추가하는 함수
def add_message(role, message):
    """새로운 대화 메시지를 세션 상태에 저장"""
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# PDF 파일을 벡터 임베딩으로 변환하는 함수 (캐시 적용으로 재처리 방지)
@st.cache_resource(show_spinner="📄 업로드된 PDF를 분석하고 있습니다...")
def embed_file(file, search_k=6):
    # 업로드된 파일을 로컬 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할 (긴 문서를 작은 청크로 나누어 검색 성능 향상)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 각 청크의 최대 문자 수
        chunk_overlap=50,  # 청크 간 겹치는 문자 수 (문맥 연결성 유지)
    )
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 단계 4: FAISS 벡터 데이터베이스 생성 (빠른 유사도 검색을 위함)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성 (질문과 관련된 문서 청크를 찾는 역할)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": search_k}  # 설정된 개수의 관련 문서 반환
    )
    return retriever


# RAG 체인을 생성하는 함수 (검색-생성 파이프라인)
def create_chain(retriever, model_name="gpt-4.1", response_length=3):
    """Retrieval-Augmented Generation 체인 생성"""
    # 단계 6: 프롬프트 템플릿 로드
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: OpenAI 언어모델 초기화 (temperature=0으로 일관된 답변 생성)
    llm = ChatOpenAI(
        model_name=model_name, temperature=0  # 창의성보다 정확성을 위해 0으로 설정
    )

    # 단계 8: RAG 체인 구성 (검색 → 프롬프트 → LLM → 출력 파싱)
    chain = (
        {
            "context": retriever,  # 관련 문서 검색
            "question": RunnablePassthrough(),  # 사용자 질문 전달
            "response_length": lambda _: response_length,  # 답변 길이 설정 전달
        }
        | prompt  # 프롬프트 템플릿 적용
        | llm  # 언어모델로 답변 생성
        | StrOutputParser()  # 문자열 형태로 결과 파싱
    )
    return chain


# PDF 파일 업로드 시 RAG 시스템 초기화
if uploaded_file:
    # 업로드된 파일을 벡터 데이터베이스로 변환 (설정된 검색 개수 적용)
    retriever = embed_file(uploaded_file, search_k=search_k)
    # 선택된 모델과 답변 길이 설정으로 RAG 체인 생성
    chain = create_chain(
        retriever, model_name=selected_model, response_length=response_length
    )
    st.session_state["chain"] = chain
    st.success(
        f"✅ '{uploaded_file.name}' 파일이 성공적으로 로드되었습니다! (검색 문서: {search_k}개)"
    )

# 대화 초기화 버튼 클릭 시
if clear_btn:
    st.session_state["messages"] = []
    st.rerun()  # 페이지 새로고침

# 이전 대화 기록 출력
print_messages()

# 사용자 질문 입력창
user_input = st.chat_input("📝 PDF 내용에 대해 궁금한 점을 물어보세요!")

# 경고 메시지 표시를 위한 빈 공간
warning_msg = st.empty()

# 사용자 질문 처리 및 답변 생성
if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 질문 표시
        st.chat_message("user").write(user_input)

        # AI 답변을 스트리밍 방식으로 실시간 표시
        with st.chat_message("assistant"):
            # 스트리밍 답변을 위한 컨테이너
            container = st.empty()
            ai_answer = ""

            # RAG 체인을 통해 스트리밍 답변 생성
            response = chain.stream(user_input)
            for token in response:
                ai_answer += token
                # 실시간으로 답변 업데이트
                container.markdown(ai_answer)

        # 대화 기록을 세션에 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # PDF 파일 미업로드 시 경고 메시지
        warning_msg.error("⚠️ 먼저 PDF 파일을 업로드해 주세요.")
