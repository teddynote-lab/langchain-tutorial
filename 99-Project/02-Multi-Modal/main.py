# Streamlit 및 기본 라이브러리
import streamlit as st
import os

# LangChain 관련 라이브러리
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.models import MultiModal
from langchain_teddynote.prompts import load_prompt

# 환경 설정
from dotenv import load_dotenv


# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
logging.langsmith("LangGraph-Tutorial")

# 캐시 디렉토리 생성 (이미지 업로드 저장을 위함)
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 이미지 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 벡터 임베딩 저장 폴더 (향후 확장성을 위해 유지)
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# Streamlit 앱 제목 설정
st.title("이미지 인식 기반 멀티모달 AI 챗봇")

# Streamlit 세션 상태 초기화 (앱 재실행 시에도 대화 기록 유지)
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 리스트 초기화
    st.session_state["messages"] = []

# 세션 상태에 현재 이미지 관리
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None


# 사이드바 UI 구성
with st.sidebar:
    # 대화 기록 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화")

    # 이미지 파일 업로드 위젯
    image_url = st.text_input(
        "이미지 파일 링크(URL)",
        help="이미지 파일의 링크를 입력하면 분석할 수 있습니다.",
    )

    # 분석 도메인 컨텍스트 선택
    domain_context = st.selectbox(
        "🎯 분석 도메인",
        ["일반 분석", "금융/재무제표", "의료/헬스케어", "기술/IT", "교육/학습"],
        index=0,
        help="이미지 분석 시 적용할 전문 도메인을 선택하세요.",
    )

    # 답변 길이 조절 슬라이더
    response_length = st.slider(
        "📏 답변 길이 설정",
        min_value=1,
        max_value=5,
        value=3,
        help="1: 간단 (1-2문장), 2: 짧음 (1문단), 3: 보통 (2-3문단), 4: 자세함 (4-5문단), 5: 매우 자세함 (5문단 이상)",
    )

    # Temperature 설정 (모델 창의성 조절)
    temperature = st.slider(
        "🌡️ Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="0에 가까울수록 정확하고 일관된 답변, 1에 가까울수록 창의적인 답변",
    )


# 이전 대화 기록을 화면에 출력하는 함수
def print_messages():
    """저장된 대화 기록을 현재 컨텍스트에 표시"""
    if st.session_state["messages"]:
        st.markdown("### 💬 대화 기록")
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)
    else:
        st.info("💭 아직 대화가 없습니다. 이미지를 업로드하고 질문을 시작해 보세요!")


# 새로운 메시지를 세션 상태에 추가하는 함수
def add_message(role, message):
    """새로운 대화 메시지를 세션 상태에 저장"""
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 외부 프롬프트를 로드하여 멀티모달 답변 생성하는 함수
def generate_answer(
    img_url,
    user_prompt,
    temperature=0.1,
    response_length=3,
    domain_context="금융/재무제표",
):
    """이미지와 텍스트를 결합한 멀티모달 AI 답변 생성"""
    # 입력값 검증
    if not img_url or not user_prompt:
        raise ValueError("이미지 주소(URL) 와 사용자 프롬프트가 필요합니다.")

    if not isinstance(user_prompt, str) or user_prompt.strip() == "":
        raise ValueError("유효한 사용자 프롬프트가 필요합니다.")

    # 외부 프롬프트 템플릿 로드
    try:
        prompt_template = load_prompt("prompts/multimodal.yaml", encoding="utf-8")
    except Exception as e:
        raise ValueError(f"프롬프트 템플릿 로드 실패: {e}")

    # 프롬프트 변수에 값 할당 (None 값 방지)
    system_prompt = prompt_template.format(
        response_length=response_length if response_length is not None else 3,
        domain_context=domain_context if domain_context is not None else "일반 분석",
    )

    # OpenAI ChatGPT 모델 초기화 (설정된 temperature 적용)
    llm = ChatOpenAI(
        model="openai/gpt-4.1",
        temperature=temperature,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )

    # LangChain 멀티모달 객체 생성 (이미지 + 텍스트 처리)
    try:
        multimodal = MultiModal(
            llm, system_prompt=system_prompt, user_prompt=user_prompt.strip()
        )

        # 이미지 파일에 대한 스트리밍 방식 질의 및 답변 생성
        answer = multimodal.stream(image_url)
        return answer
    except Exception as e:
        raise RuntimeError(f"멀티모달 처리 중 오류 발생: {e}")


# 대화 초기화 버튼 클릭 시
if clear_btn:
    st.session_state["messages"] = []
    # st.session_state["current_image"] = None  # 이미지는 유지
    st.rerun()  # 페이지 새로고침

# 업로드된 파일 처리
if image_url:
    st.session_state["current_image"] = image_url

# 현재 이미지 미리보기
if st.session_state["current_image"]:
    st.subheader("🖼️ 업로드된 이미지")
    st.image(st.session_state["current_image"], use_container_width=True)
    st.divider()

# AI 어시스턴트 영역
st.subheader("💬 AI 어시스턴트")

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("📝 이미지 내용에 대해 궁금한 점을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 현재 이미지가 있는지 확인
    if st.session_state["current_image"]:
        # 현재 이미지 사용
        img = st.session_state["current_image"]

        try:
            # 답변 요청 (새로운 구성 옵션들을 포함)
            response = generate_answer(
                img_url=img,
                user_prompt=user_input,
                temperature=temperature,
                response_length=response_length,
                domain_context=domain_context,
            )
        except Exception as e:
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")
            st.info("💡 다시 시도해 주시거나, 다른 이미지나 질문을 사용해 보세요.")
            response = None

        # 응답이 성공적으로 생성된 경우에만 채팅 표시
        if response is not None:
            # 사용자의 입력 표시
            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                ai_answer = st.write_stream(response)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)
    else:
        # 이미지 파일 미업로드 시 경고 메시지
        st.error("⚠️ 먼저 이미지 파일을 업로드해 주세요.")
