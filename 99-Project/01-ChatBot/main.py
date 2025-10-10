import os
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging

# 환경 설정
from dotenv import load_dotenv

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
logging.langsmith("LangGraph-Tutorial")

# Streamlit 앱 제목 설정
st.title("💬 AI 챗봇")

# Streamlit 세션 상태 초기화 (앱 재실행 시에도 대화 기록 유지)
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 리스트 초기화
    st.session_state["messages"] = []

# 사이드바 UI 구성
with st.sidebar:
    # 대화 기록 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화")

    # Temperature 설정 (모델 창의성 조절)
    temperature = st.slider(
        "🌡️ Temperature (창의성)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="0에 가까울수록 정확하고 일관된 답변, 2에 가까울수록 창의적이고 다양한 답변",
    )

    # 답변 길이 조절 선택박스
    response_length = st.selectbox(
        "📏 답변 길이 설정",
        ["간단", "보통", "자세함", "매우 자세함"],
        index=1,
        help="AI 답변의 길이를 조절합니다.",
    )

    st.divider()

    # 시스템 프롬프트 설정
    st.subheader("⚙️ 시스템 프롬프트")
    system_prompt = st.text_area(
        "AI의 역할과 성격을 정의하세요",
        value="당신은 도움이 되고 친근한 AI 어시스턴트입니다. 사용자의 질문에 정확하고 유용한 답변을 제공하며, 필요시 추가 정보나 예시를 포함하여 설명합니다.",
        height=150,
        help="AI의 역할, 성격, 답변 스타일 등을 정의할 수 있습니다.",
    )


# 이전 대화 기록을 화면에 출력하는 함수
def print_messages():
    """저장된 대화 기록을 순서대로 화면에 표시"""
    if st.session_state["messages"]:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)
    else:
        st.info("💭 안녕하세요! 무엇이든 물어보세요.")


# 새로운 메시지를 세션 상태에 추가하는 함수
def add_message(role, message):
    """새로운 대화 메시지를 세션 상태에 저장"""
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 답변 길이에 따른 지시사항 생성
def get_length_instruction(length):
    """선택된 답변 길이에 따른 지시사항 반환"""
    length_map = {
        "간단": "간단하고 핵심적인 답변을 1-2문장으로 제공하세요.",
        "보통": "적절한 길이로 명확하게 답변하세요. (2-3문단 정도)",
        "자세함": "상세하고 포괄적인 답변을 제공하세요. 예시나 추가 설명을 포함하세요.",
        "매우 자세함": "매우 상세하고 심층적인 답변을 제공하세요. 다양한 관점과 예시, 관련 정보를 포함하세요.",
    }
    return length_map.get(length, "적절한 길이로 답변하세요.")


# AI 답변 생성 함수
def generate_answer(user_input, system_prompt, temperature, response_length):
    """사용자 입력에 대한 AI 답변 생성"""
    try:
        # OpenAI 모델 초기화
        llm = ChatOpenAI(
            model="openai/gpt-4.1",
            temperature=temperature,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
        )

        # 시스템 프롬프트에 답변 길이 지시사항 추가
        length_instruction = get_length_instruction(response_length)
        enhanced_system_prompt = f"{system_prompt}\n\n답변 스타일: {length_instruction}"

        # 메시지 구성
        messages = [
            SystemMessage(content=enhanced_system_prompt),
            HumanMessage(content=user_input),
        ]

        # 스트리밍 응답 생성
        response = llm.stream(messages)
        return response

    except Exception as e:
        raise RuntimeError(f"답변 생성 중 오류 발생: {e}")


# 대화 초기화 버튼 클릭 시
if clear_btn:
    st.session_state["messages"] = []
    st.rerun()

# 이전 대화 기록 출력
print_messages()

# 사용자 질문 입력창
user_input = st.chat_input("💬 무엇이든 물어보세요!")

# 사용자 질문 처리 및 답변 생성
if user_input:
    try:
        # AI 답변 생성
        response = generate_answer(
            user_input=user_input,
            system_prompt=system_prompt,
            temperature=temperature,
            response_length=response_length,
        )

        # 사용자 질문 표시
        st.chat_message("user").write(user_input)

        # AI 답변을 스트리밍 방식으로 실시간 표시
        with st.chat_message("assistant"):
            ai_answer = st.write_stream(response)

        # 대화 기록을 세션에 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {str(e)}")
        st.info("💡 다시 시도해 주시거나, 모델 설정을 확인해 보세요.")

# 사이드바 하단에 현재 설정 정보 표시
with st.sidebar:
    st.divider()
    st.markdown("### 📊 현재 설정")
    st.caption(f"**Temperature:** {temperature}")
    st.caption(f"**답변 길이:** {response_length}")
