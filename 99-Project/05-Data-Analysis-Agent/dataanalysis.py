from langchain.tools import tool
from typing import Annotated, Optional
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)


class DataAnalysisAgent:
    """
    LangChain을 활용한 데이터 분석 에이전트 클래스

    CSV, Excel 파일을 업로드하여 pandas, matplotlib, seaborn을 이용한
    데이터 분석 및 시각화를 수행하는 AI 에이전트입니다.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4.1",  # 기본 모델을 gpt-4.1로 설정
        prefix_prompt: Optional[str] = None,
        postfix_prompt: Optional[str] = None,
        column_guideline: Optional[str] = None,
    ):
        """
        Initialize the DataAnalysisAgent with a dataframe and configuration.

        Args:
            dataframe (pd.DataFrame): The dataframe to analyze
            model_name (str): OpenAI model to use for analysis
            prefix_prompt (Optional[str]): Additional prompt to prepend
            postfix_prompt (Optional[str]): Additional prompt to append
            column_guideline (Optional[str]): User-provided column descriptions
        """
        # 기본 속성 설정
        self.df = dataframe  # 분석할 데이터프레임
        self.model_name = model_name  # 사용할 OpenAI 모델
        self.prefix_prompt = prefix_prompt  # 추가 프롬프트 (앞부분)
        self.postfix_prompt = postfix_prompt  # 추가 프롬프트 (뒷부분)

        # 컬럼 가이드라인 설정
        if column_guideline is not None and column_guideline.strip() != "":
            COLUMN_GUIDE_PREFIX = "###\n\n# Column Guideline\n\nHere's the column guideline you'll be working with:\n"
            self.column_guideline = COLUMN_GUIDE_PREFIX + column_guideline
        else:
            self.column_guideline = ""

        # 도구 및 저장소 초기화
        self.tools = [self.create_python_repl_tool()]  # Python 실행 도구 생성
        self.store = {}  # 채팅 히스토리 저장소

        # 에이전트 설정 및 초기화
        self.setup_agent()

    def create_python_repl_tool(self):
        """
        Python 코드 실행 도구를 생성하는 메서드
        pandas, matplotlib, seaborn을 사용한 데이터 분석 및 시각화 코드를 실행합니다.

        Returns:
            tool: LangChain 도구 객체
        """

        @tool
        def python_repl_tool(
            code: Annotated[str, "Any python code(pandas, matplotlib, seaborn) to run"],
        ):
            """Use this tool to run python, pandas query, matplotlib, and seaborn code."""
            try:
                # Python AST REPL 도구 생성 (보안상 AST 사용)
                python_tool = PythonAstREPLTool(
                    locals={
                        "df": self.df,  # 분석 대상 데이터프레임
                        "sns": sns,  # seaborn 시각화 라이브러리
                        "plt": plt,  # matplotlib 시각화 라이브러리
                        "pd": pd,  # pandas 데이터 처리 라이브러리
                    }
                )
                # 코드 실행 및 결과 반환
                return python_tool.invoke(code)
            except Exception as e:
                # 실행 오류 발생 시 오류 메시지 반환
                return f"Execution failed. Error: {repr(e)}"

        return python_repl_tool

    def build_system_prompt(self):
        """
        시스템 프롬프트를 구성하는 메서드
        YAML 파일에서 프롬프트를 로드하고 데이터프레임 정보와 컬럼 가이드라인을 추가합니다.

        Returns:
            str: 완성된 시스템 프롬프트
        """
        # YAML 파일에서 프롬프트 템플릿 로드
        system_prompt = load_prompt("prompts/data-analysis.yaml", encoding="utf-8")

        # 데이터프레임 정보와 컬럼 가이드라인을 프롬프트에 삽입
        system_prompt = system_prompt.format(
            dataframe_head=self.df.head().to_string(),  # 데이터프레임 상위 5개 행
            column_guideline=self.column_guideline,  # 사용자 정의 컬럼 설명
        )

        # 사용자가 지정한 추가 프롬프트가 있는 경우 앞부분에 추가
        if self.prefix_prompt is not None:
            system_prompt = f"{self.prefix_prompt}\n\n{system_prompt}"

        # 사용자가 지정한 추가 프롬프트가 있는 경우 뒷부분에 추가
        if self.postfix_prompt is not None:
            system_prompt = f"{system_prompt}\n\n{self.postfix_prompt}"

        return system_prompt

    def setup_agent(self):
        """
        LangChain 에이전트를 설정하는 메서드
        프롬프트 템플릿, LLM 모델, 도구들을 조합하여 에이전트를 생성합니다.
        """
        # 채팅 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.build_system_prompt(),  # 시스템 프롬프트 설정
                ),
                ("placeholder", "{chat_history}"),  # 채팅 히스토리 자리표시자
                ("human", "{input}"),  # 사용자 입력 자리표시자
                ("placeholder", "{agent_scratchpad}"),  # 에이전트 작업 공간 자리표시자
            ]
        )

        # OpenAI LLM 모델 초기화 (temperature=0으로 일관된 결과 보장)
        llm = ChatOpenAI(model=self.model_name, temperature=0)

        # 도구 호출이 가능한 에이전트 생성
        agent = create_tool_calling_agent(llm, self.tools, prompt)

        # 에이전트 실행기 생성 및 설정
        self.agent_executor = AgentExecutor(
            agent=agent,  # 생성된 에이전트
            tools=self.tools,  # 사용 가능한 도구 목록
            verbose=True,  # 상세 로그 출력
            max_iterations=20,  # 최대 반복 횟수
            max_execution_time=60,  # 최대 실행 시간 (초)
            handle_parsing_errors=True,  # 파싱 오류 처리 활성화
        )

    def get_session_history(self, session_id):
        """
        세션 ID에 따른 채팅 히스토리를 가져오는 메서드
        새로운 세션 ID인 경우 새로운 ChatMessageHistory 객체를 생성합니다.

        Args:
            session_id (str): 세션 식별자

        Returns:
            ChatMessageHistory: 해당 세션의 채팅 히스토리 객체
        """
        return self.store.setdefault(session_id, ChatMessageHistory())

    def get_agent_with_chat_history(self):
        """
        채팅 히스토리를 포함한 에이전트를 반환하는 메서드
        RunnableWithMessageHistory를 사용하여 대화 기록을 유지합니다.

        Returns:
            RunnableWithMessageHistory: 채팅 히스토리가 포함된 실행 가능한 에이전트
        """
        return RunnableWithMessageHistory(
            self.agent_executor,  # 에이전트 실행기
            self.get_session_history,  # 세션 히스토리 가져오기 함수
            input_messages_key="input",  # 입력 메시지 키
            history_messages_key="chat_history",  # 히스토리 메시지 키
        )

    def stream(self, input_query, session_id="data_analysis_session"):
        """
        사용자 질문에 대한 스트리밍 응답을 생성하는 메서드
        실시간으로 응답을 받을 수 있어 사용자 경험을 향상시킵니다.

        Args:
            input_query (str): 사용자의 질문/요청
            session_id (str): 대화 세션 식별자 (기본값: "data_analysis_session")

        Returns:
            Iterator: 스트리밍 응답 이터레이터
        """
        # 채팅 히스토리가 포함된 에이전트 가져오기
        agent_with_chat_history = self.get_agent_with_chat_history()

        # 스트리밍 응답 생성
        response = agent_with_chat_history.stream(
            {"input": input_query},  # 사용자 입력
            config={"configurable": {"session_id": session_id}},  # 세션 설정
        )
        return response
