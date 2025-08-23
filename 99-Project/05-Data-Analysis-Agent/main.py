from typing import List, Union
from langchain_experimental.tools import PythonAstREPLTool
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-Tutorial")


##### 폰트 설정 #####
def setup_korean_font():
    """한글 폰트를 설정하는 함수"""
    current_os = platform.system()

    if current_os == "Windows":
        # Windows 환경 폰트 설정 (맑은 고딕)
        font_path = "C:/Windows/Fonts/malgun.ttf"
        fontprop = fm.FontProperties(fname=font_path, size=12)
        plt.rc("font", family=fontprop.get_name())
    elif current_os == "Darwin":  # macOS
        # Mac 환경 폰트 설정 (애플 고딕)
        plt.rcParams["font.family"] = "AppleGothic"
    else:  # Linux 등 기타 OS
        # 기본 한글 폰트 설정 시도 (나눔 고딕)
        try:
            plt.rcParams["font.family"] = "NanumGothic"
        except:
            print("한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")

    # 마이너스 폰트 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False


# 폰트 설정 실행
setup_korean_font()

# Streamlit 앱 설정
st.title("📊 데이터 분석 에이전트")
st.markdown("### LangChain을 활용한 지능형 데이터 분석 챗봇")

# 데이터 미리보기 섹션 (항상 표시)
with st.expander("📋 데이터 미리보기", expanded=True):
    if "df" in st.session_state and st.session_state["df"] is not None:
        loaded_data = st.session_state["df"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 데이터 기본 정보**")
            st.write(f"• 행 수: {loaded_data.shape[0]:,}개")
            st.write(f"• 열 수: {loaded_data.shape[1]}개")
            if st.session_state.get("filename"):
                st.write(f"• 파일명: {st.session_state['filename']}")
        
        with col2:
            st.markdown("**📁 컬럼 목록**")
            for col in loaded_data.columns:
                st.write(f"• {col}")
        
        st.markdown("**🔍 상위 5개 행**")
        st.dataframe(loaded_data.head(), use_container_width=True)
    else:
        st.info("📥 사이드바에서 파일을 업로드하고 '데이터 분석 시작'을 클릭하면 데이터 미리보기가 표시됩니다.")

st.divider()

# 세션 상태 초기화
def init_session_state():
    """세션 상태를 초기화하는 함수"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # 대화 내용을 저장할 리스트 초기화
    if "df" not in st.session_state:
        st.session_state["df"] = None  # 데이터프레임 저장
    if "agent" not in st.session_state:
        st.session_state["agent"] = None  # 에이전트 저장
    if "python_tool" not in st.session_state:
        st.session_state["python_tool"] = None  # Python 도구 저장
    if "filename" not in st.session_state:
        st.session_state["filename"] = None  # 파일명 저장


# 세션 상태 초기화 실행
init_session_state()


# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지


# 메시지 관련 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role, avatar="💻"):
            for content in content_list:
                if isinstance(content, list):
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # 대화 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화", use_container_width=True)

    st.divider()

    # 파일 업로드 섹션
    st.subheader("📁 파일 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드해주세요.",
        type=["csv", "xlsx", "xls"],
        help="분석할 데이터 파일을 선택해주세요. (CSV, Excel 형식 지원)",
    )

    # 모델 선택 섹션
    st.subheader("🤖 AI 모델 설정")
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.",
        ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
        index=0,
        help="분석에 사용할 AI 모델을 선택해주세요.",
    )

    # 차트 스타일 설정 섹션
    st.subheader("🎨 차트 스타일 설정")
    chart_style = st.selectbox(
        "차트 스타일을 선택해주세요.",
        ["default", "whitegrid", "darkgrid", "white", "dark", "ticks"],
        index=1,
        help="matplotlib 및 seaborn 차트의 기본 스타일을 설정합니다.",
    )

    color_palette = st.selectbox(
        "컬러 팔레트를 선택해주세요.",
        ["husl", "deep", "muted", "pastel", "bright", "dark", "colorblind"],
        index=0,
        help="차트에 사용할 컬러 팔레트를 설정합니다.",
    )

    st.divider()

    # 컬럼 가이드라인 섹션
    st.subheader("📋 컬럼 설명")
    user_column_guideline = st.text_area(
        "컬럼에 대한 설명을 입력해주세요.",
        placeholder="예: age - 사용자의 나이\nsalary - 월급여(만원 단위)\ndepartment - 소속 부서",
        height=150,
        help="데이터의 컬럼에 대한 상세한 설명을 입력하면 더 정확한 분석이 가능합니다.",
    )

    # 분석 시작 버튼
    apply_btn = st.button(
        "🚀 데이터 분석 시작", use_container_width=True, type="primary"
    )

    st.divider()

    # 현재 설정된 컬럼 가이드라인 표시
    txt_column_guideline = st.empty()


# 차트 스타일 적용 함수
def apply_chart_style(chart_style, color_palette):
    """
    선택된 차트 스타일과 컬러 팔레트를 적용하는 함수

    Args:
        chart_style (str): 선택된 차트 스타일
        color_palette (str): 선택된 컬러 팔레트
    """
    try:
        # seaborn 스타일 적용
        if chart_style != "default":
            sns.set_style(chart_style)

        # 컬러 팔레트 적용
        sns.set_palette(color_palette)
    except Exception as e:
        st.warning(f"차트 스타일 적용 중 오류가 발생했습니다: {e}")


# 파일 로드 함수
def load_data_file(uploaded_file):
    """
    업로드된 파일을 로드하는 함수

    Args:
        uploaded_file: Streamlit의 업로드된 파일 객체

    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "csv":
            # CSV 파일 로드 (인코딩 자동 감지)
            return pd.read_csv(uploaded_file, encoding="utf-8")
        elif file_extension in ["xlsx", "xls"]:
            # Excel 파일 로드
            return pd.read_excel(uploaded_file)
        else:
            st.error(f"지원하지 않는 파일 형식입니다: {file_extension}")
            return None

    except UnicodeDecodeError:
        # UTF-8로 실패한 경우 다른 인코딩으로 시도
        try:
            return pd.read_csv(uploaded_file, encoding="cp949")
        except:
            return pd.read_csv(uploaded_file, encoding="euc-kr")
    except Exception as e:
        st.error(f"파일 로드 중 오류가 발생했습니다: {e}")
        return None


# 콜백 함수
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수

    Args:
        tool (dict): 실행된 도구 정보
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_tool":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("code")
            if query:
                df_in_result = None
                with st.status("💡 데이터 분석 실행 중...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])

                    # 차트 스타일 적용 (차트 관련 코드인 경우)
                    if any(
                        keyword in query
                        for keyword in ["plt.", "sns.", "plot", "chart"]
                    ):
                        apply_chart_style(chart_style, color_palette)

                    if "df" in st.session_state and st.session_state["df"] is not None:
                        result = st.session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(
                        label="✅ 실행 완료", state="complete", expanded=False
                    )

                # 데이터프레임 결과가 있는 경우 표시
                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                # 차트가 생성된 경우 표시
                if "plt.show" in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                    plt.close()  # 메모리 정리

                return result
            else:
                st.error(
                    "❌ 데이터프레임이 정의되지 않았습니다. 파일을 먼저 업로드해주세요."
                )
                return


def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][
                1
            ].clear()  # 에러 발생 시 마지막 메시지 삭제


def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    pass  # 현재는 아무 동작도 하지 않습니다


# 에이전트 생성 함수
def create_agent(
    dataframe,
    selected_model="gpt-4.1",
    prefix_prompt=None,
    postfix_prompt=None,
    user_column_guideline=None,
):
    """
    데이터 분석 에이전트를 생성하는 함수

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4.1"
        prefix_prompt (str, optional): 추가할 시스템 프롬프트 (앞부분)
        postfix_prompt (str, optional): 추가할 시스템 프롬프트 (뒷부분)
        user_column_guideline (str, optional): 사용자가 입력한 컬럼 가이드라인

    Returns:
        DataAnalysisAgent: 생성된 데이터 분석 에이전트
    """
    from dataanalysis import DataAnalysisAgent

    return DataAnalysisAgent(
        dataframe,
        model_name=selected_model,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,
        column_guideline=user_column_guideline,
    )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 AI 에이전트로부터 응답을 받는 함수

    Args:
        query (str): 사용자의 질문
    """
    if "agent" in st.session_state and st.session_state["agent"] is not None:
        # 사용자 메시지 표시
        st.chat_message("user", avatar="🧑‍💻").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        # 에이전트로부터 응답 받기
        agent = st.session_state["agent"]
        response = agent.stream(query, "data_analysis_session")

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant", avatar="💻"):
            has_dataframe = False
            has_chart = False

            # 응답 스트림 처리
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]

                # DataFrame 또는 차트가 출력되었는지 확인
                if (
                    st.session_state["messages"]
                    and st.session_state["messages"][-1][0] == MessageRole.ASSISTANT
                ):
                    for content in st.session_state["messages"][-1][1]:
                        if isinstance(content, list):
                            if content[0] == MessageType.DATAFRAME:
                                has_dataframe = True
                            elif content[0] == MessageType.FIGURE:
                                has_chart = True

            # 결과에 따라 적절한 응답 표시
            if has_dataframe and not has_chart:
                # DataFrame만 있는 경우 (조회 질문)
                pass  # DataFrame은 이미 tool_callback에서 표시됨
            elif has_chart and not ai_answer.strip():
                # 시각화만 있는 경우
                pass  # 차트는 이미 tool_callback에서 표시됨
            elif ai_answer.strip() and not has_dataframe and not has_chart:
                # 일반적인 텍스트 답변 (EDA, 분석 결과 등)
                st.write(ai_answer)
                add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])
            elif ai_answer.strip():
                # 텍스트 답변과 함께 DataFrame/차트가 있는 경우
                st.write(ai_answer)
                add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])
    else:
        st.error(
            "❌ 에이전트가 초기화되지 않았습니다. 먼저 파일을 업로드하고 '데이터 분석 시작'을 클릭해주세요."
        )


# 메인 로직
if clear_btn:
    # 대화 내용 및 관련 세션 상태 초기화
    st.session_state["messages"] = []
    # 데이터 관련 세션도 함께 초기화 (선택적)
    # st.session_state["df"] = None
    # st.session_state["filename"] = None
    # st.session_state["agent"] = None
    # st.session_state["python_tool"] = None
    st.rerun()

if apply_btn and uploaded_file:
    with st.spinner("📊 데이터 로딩 중..."):
        # 파일 로드
        loaded_data = load_data_file(uploaded_file)

        if loaded_data is not None:
            # 세션 상태에 데이터 저장
            st.session_state["df"] = loaded_data
            st.session_state["filename"] = uploaded_file.name  # 파일명 저장
            st.session_state["python_tool"] = PythonAstREPLTool()
            st.session_state["python_tool"].locals["df"] = loaded_data

            # 에이전트 생성
            st.session_state["agent"] = create_agent(
                loaded_data,
                selected_model,
                prefix_prompt=None,
                postfix_prompt=None,
                user_column_guideline=user_column_guideline,
            )

            # 성공 메시지
            st.success("✅ 데이터 분석 준비가 완료되었습니다!")
            st.info("💬 이제 아래 채팅창에 질문을 입력해주세요!")

elif apply_btn:
    st.warning("⚠️ 분석할 파일을 먼저 업로드해주세요.")

# 현재 설정된 컬럼 가이드라인 표시 (에이전트가 생성된 경우)
if "agent" in st.session_state and st.session_state["agent"] is not None:
    if st.session_state["agent"].column_guideline:
        txt_column_guideline.markdown(
            f"**📋 현재 적용된 컬럼 설명**\n\n```\n{st.session_state['agent'].column_guideline}\n```"
        )

# 저장된 메시지 출력
print_messages()

# 사용자 입력 받기
user_input = st.chat_input(
    "💭 데이터에 대해 궁금한 내용을 물어보세요! (예: 데이터를 요약해줘, 상관관계를 시각화해줘)"
)
if user_input:
    ask(user_input)
