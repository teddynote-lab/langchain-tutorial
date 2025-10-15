import os
import sys
import uuid
from typing import Any, Dict, List

import streamlit as st

# 환경 설정
from dotenv import load_dotenv

# LangChain 관련 라이브러리
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate

# MCP 클라이언트
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
logging.langsmith("LangChain-Tutorial")

# Windows MCP stdio stderr 버그 워크어라운드
# 이슈: https://github.com/modelcontextprotocol/python-sdk/issues/1103
# Windows에서 sys.stderr를 사용할 때 발생하는 문제를 해결하기 위한 패치
print("=== Windows MCP stdio 패치 적용 ===\n")

try:
    # MCP SDK의 stdio_client 함수를 패치
    from contextlib import asynccontextmanager

    import mcp.client.stdio as stdio_module
    from mcp.client.stdio import StdioServerParameters

    # 원본 stdio_client 함수 백업
    _original_stdio_client = stdio_module.stdio_client

    # errlog=None을 강제하는 래퍼 함수 생성
    @asynccontextmanager
    async def patched_stdio_client(server, **kwargs):
        """Windows에서 stderr 문제를 우회하기 위해 errlog=None을 강제"""
        # errlog 인자를 제거하고 None으로 설정
        kwargs["errlog"] = None
        async with _original_stdio_client(server, **kwargs) as streams:
            yield streams

    # 원본 함수를 패치된 버전으로 교체
    stdio_module.stdio_client = patched_stdio_client

    print("✅ Windows MCP stdio 패치가 성공적으로 적용되었습니다.")
    print("   - mcp.client.stdio.stdio_client 함수가 패치되었습니다.")
    print("   - errlog=None이 자동으로 적용됩니다.\n")

except Exception as e:
    print(f"⚠️ MCP 패치 적용 실패: {e}")
    print(f"   타입: {type(e).__name__}")
    print("   기본 설정으로 계속 진행합니다.\n")

# Streamlit 앱 제목 설정
st.title("MCP Agent 챗봇")

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
if "current_mcp_config" not in st.session_state:
    # MCP 서버 설정 변경 감지를 위한 상태
    st.session_state["current_mcp_config"] = None
if "custom_prompt" not in st.session_state:
    st.session_state["custom_prompt"] = (
        "당신은 스마트 MCP Agent입니다. 주어진 MCP 도구를 활용하여 사용자의 질문에 응답하세요.\n"
        "문제를 해결하기 위해 다양한 MCP 도구를 사용할 수 있습니다.\n"
        "답변은 친근감 있는 어조로 답변하세요."
    )
# Smithery 서버 설정
if "smithery_servers" not in st.session_state:
    st.session_state["smithery_servers"] = []


async def setup_mcp_client(server_configs: Dict[str, Any]):
    """
    MCP 클라이언트를 설정하고 도구를 가져옵니다.

    Args:
        server_configs: 서버 설정 정보 딕셔너리

    Returns:
        tuple: (MCP 클라이언트, 도구 리스트)
    """
    # MultiServerMCPClient 인스턴스 생성 - 여러 MCP 서버를 통합 관리
    client = MultiServerMCPClient(server_configs)

    # 모든 연결된 서버로부터 사용 가능한 도구들을 수집
    tools = await client.get_tools()

    # 로드된 도구 정보를 콘솔에 출력 (디버깅 및 확인용)
    print(f"✅ {len(tools)} 개의 MCP 도구가 로드되었습니다:")
    for tool in tools:
        print(f"  - {tool.name}")

    return client, tools


async def create_mcp_react_agent(
    server_configs: Dict[str, Any],
    model_name: str,
    temperature: float,
    custom_prompt: str,
):
    """
    MCP 도구를 사용하는 React Agent를 생성합니다.

    Args:
        server_configs: MCP 서버 설정 딕셔너리
        model_name: 사용할 LLM 모델 이름
        temperature: 모델 temperature 설정
        custom_prompt: 커스텀 시스템 프롬프트

    Returns:
        LangGraph 에이전트: MCP 도구가 연결된 React Agent
    """
    # MCP 클라이언트 생성 및 모든 서버의 도구를 통합 로딩
    client, tools = await setup_mcp_client(server_configs=server_configs)

    # OpenRouter를 통한 GPT 모델 사용
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )

    # 기존 메모리 인스턴스 사용 (대화 기록 유지)
    memory = st.session_state["memory"]

    # 커스텀 프롬프트 설정
    if custom_prompt:
        # 사용자 메시지와 함께 시스템 프롬프트 적용
        system_prompt = ChatPromptTemplate.from_messages(
            [("system", custom_prompt), ("placeholder", "{messages}")]
        )

        # React Agent 생성 (커스텀 프롬프트 적용)
        agent = create_react_agent(
            llm, tools, checkpointer=memory, prompt=system_prompt
        )
    else:
        # React Agent 생성 (기본 프롬프트)
        agent = create_react_agent(llm, tools, checkpointer=memory)

    return agent, tools


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
                        with st.expander(f"🛠️ MCP 도구 호출 정보", expanded=False):
                            for i, tool_call in enumerate(tool_calls, 1):
                                st.markdown(f"**{i}. {tool_call['name']}**")

                                # 도구 호출 인자 표시
                                if tool_call["args"]:
                                    st.markdown("📝 **호출 인자**")
                                    for key, value in tool_call["args"].items():
                                        # 값이 너무 긴 경우 축약
                                        if isinstance(value, str) and len(value) > 100:
                                            value = value[:100] + "..."
                                        st.markdown(f"  • `{key}`: {value}")

                                # 도구 실행 결과 표시
                                if "result" in tool_call:
                                    st.markdown("📊 **실행 결과**")
                                    st.write(tool_call["result"])

                                if i < len(tool_calls):
                                    st.divider()

                    # AI 응답 표시
                    st.markdown(content)

            # 기존 ChatMessage 형태인 경우 (하위 호환성)
            else:
                st.chat_message(msg_data.role).write(msg_data.content)
    else:
        st.info(
            "💭 안녕하세요! MCP Agent와 대화해보세요. 다양한 MCP 도구를 활용할 수 있습니다."
        )


def add_message(role: str, message: str, tool_calls: list = None):
    """새로운 대화 메시지를 세션 상태에 저장 (도구 호출 정보 포함)"""
    msg_data = {"role": role, "content": message, "tool_calls": tool_calls or []}
    st.session_state["messages"].append(msg_data)


# 사이드바 UI 구성
with st.sidebar:
    st.header("⚙️ MCP Agent 설정")

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
        [
            "openai/gpt-4.1",
            "openai/gpt-oss-120b",
            "anthropic/claude-opus-4.1",
            "qwen/qwen3-235b-a22b-thinking-2507",
            "google/gemini-2.5-flash",
        ],
        index=0,
        help="사용할 언어모델을 선택하세요.",
    )

    temperature = st.slider(
        "🌡️ Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="0에 가까울수록 정확하고 일관된 답변, 1에 가까울수록 창의적인 답변",
    )

    st.divider()

    # 커스텀 프롬프트 설정
    st.subheader("✍️ 프롬프트 설정")
    custom_prompt = st.text_area(
        "Agent 프롬프트 편집",
        value=st.session_state["custom_prompt"],
        height=100,
        help="Agent의 역할과 행동을 정의하는 프롬프트를 수정할 수 있습니다.",
    )
    st.session_state["custom_prompt"] = custom_prompt

    st.divider()

    # MCP 서버 설정
    st.subheader("🌐 MCP 서버 설정")

    # 로컬 날씨 서버
    use_weather_server = st.checkbox(
        "🌤️ 날씨 서버 (로컬 stdio)",
        value=True,
        help="로컬 날씨 정보를 제공하는 MCP 서버",
    )

    # 원격 시간 서버
    use_time_server = st.checkbox(
        "⏰ 시간 서버 (HTTP)",
        value=False,
        help="현재 시간 정보를 제공하는 원격 MCP 서버",
    )
    time_server_url = st.text_input(
        "시간 서버 URL",
        value="http://127.0.0.1:8002/mcp",
        help="시간 서버의 HTTP 엔드포인트 URL",
    )

    # RAG 서버
    use_rag_server = st.checkbox(
        "🔍 RAG 검색 서버 (HTTP)",
        value=False,
        help="문서 검색을 위한 RAG MCP 서버",
    )
    rag_server_url = st.text_input(
        "RAG 서버 URL",
        value="http://127.0.0.1:8005/mcp",
        help="RAG 서버의 HTTP 엔드포인트 URL",
    )

    st.divider()

    # Smithery 외부 서버 설정
    st.subheader("🔌 Smithery 3rd Party 서버")

    st.markdown("**등록된 Smithery 서버:**")

    # 등록된 서버 목록 표시
    if st.session_state["smithery_servers"]:
        for idx, server in enumerate(st.session_state["smithery_servers"]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📦 {server['name']}: `{server['package']}`")
            with col2:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state["smithery_servers"].pop(idx)
                    st.rerun()
    else:
        st.caption("_등록된 서버가 없습니다._")

    # 새 Smithery 서버 추가
    with st.expander("➕ 새 Smithery 서버 추가", expanded=False):
        new_server_name = st.text_input(
            "서버 이름",
            placeholder="예: desktop-commander",
            help="MCP 설정에서 사용할 서버 식별 이름",
        )
        new_server_package = st.text_input(
            "NPM 패키지 경로",
            placeholder="예: @wonderwhy-er/desktop-commander",
            help="Smithery의 NPM 패키지 경로 (@ 포함)",
        )
        new_server_key = st.text_input(
            "API Key (선택사항)",
            type="password",
            placeholder="Smithery API Key (필요시)",
            help="일부 서비스는 API Key가 필요할 수 있습니다.",
        )

        if st.button("✅ 서버 추가"):
            if new_server_name and new_server_package:
                st.session_state["smithery_servers"].append(
                    {
                        "name": new_server_name,
                        "package": new_server_package,
                        "key": new_server_key if new_server_key else None,
                    }
                )
                st.success(f"✅ {new_server_name} 서버가 추가되었습니다!")
                st.rerun()
            else:
                st.error("⚠️ 서버 이름과 패키지 경로를 모두 입력해주세요.")

    st.divider()

    # 프리셋 Smithery 서버 추가 (빠른 설정)
    st.subheader("⚡ 프리셋 서버")
    st.caption("자주 사용되는 Smithery 서버를 빠르게 추가할 수 있습니다.")

    preset_servers = {
        "desktop-commander": {
            "package": "@wonderwhy-er/desktop-commander",
            "description": "데스크톱 파일 시스템 관리",
        },
    }

    for preset_name, preset_info in preset_servers.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"**{preset_name}**")
            st.caption(f"_{preset_info['description']}_")
        with col2:
            # 이미 추가된 서버인지 확인
            already_added = any(
                s["name"] == preset_name for s in st.session_state["smithery_servers"]
            )
            if st.button(
                "✅" if already_added else "➕",
                key=f"preset_{preset_name}",
                disabled=already_added,
            ):
                st.session_state["smithery_servers"].append(
                    {
                        "name": preset_name,
                        "package": preset_info["package"],
                        "key": None,
                    }
                )
                st.success(f"✅ {preset_name} 추가됨!")
                st.rerun()


# MCP Agent 설정 및 생성
async def setup_mcp_agent():
    """선택된 MCP 서버들을 기반으로 Agent 설정"""
    # 현재 MCP 설정을 문자열로 생성 (변경 감지용)
    current_config = {
        "weather": use_weather_server,
        "time": use_time_server,
        "time_url": time_server_url if use_time_server else None,
        "rag": use_rag_server,
        "rag_url": rag_server_url if use_rag_server else None,
        "smithery": [s["name"] for s in st.session_state["smithery_servers"]],
        "model": selected_model,
        "temperature": temperature,
        "custom_prompt": custom_prompt,
    }
    config_str = str(sorted(current_config.items()))

    # 설정이 변경된 경우에만 Agent 재생성
    if (
        st.session_state["current_mcp_config"] != config_str
        or st.session_state["agent"] is None
    ):
        server_configs = {}

        # 로컬 날씨 서버 추가
        if use_weather_server:
            server_configs["weather"] = {
                "command": "uv",
                "args": ["run", "python", "../../05-MCP/server/mcp_server_local.py"],
                "transport": "stdio",
            }

        # 원격 시간 서버 추가
        if use_time_server:
            server_configs["current_time"] = {
                "url": time_server_url,
                "transport": "streamable_http",
            }

        # RAG 서버 추가
        if use_rag_server:
            server_configs["rag_mcp"] = {
                "url": rag_server_url,
                "transport": "streamable_http",
            }

        # Smithery 서버들 추가
        for smithery_server in st.session_state["smithery_servers"]:
            args = [
                "-y",
                "@smithery/cli@latest",
                "run",
                smithery_server["package"],
            ]
            # API Key가 있으면 추가
            if smithery_server.get("key"):
                args.extend(["--key", smithery_server["key"]])

            server_configs[smithery_server["name"]] = {
                "command": "npx",
                "args": args,
                "transport": "stdio",
            }

        # Agent 생성 (서버가 하나 이상 있을 때만)
        if server_configs:
            agent, tools = await create_mcp_react_agent(
                server_configs=server_configs,
                model_name=selected_model,
                temperature=temperature,
                custom_prompt=custom_prompt,
            )
            # 설정 업데이트
            st.session_state["current_mcp_config"] = config_str
            return agent, tools
        else:
            st.session_state["current_mcp_config"] = config_str
            return None, []

    # 설정이 변경되지 않았으면 기존 Agent 사용
    else:
        return st.session_state["agent"], st.session_state["tools"]


# Agent 설정 (비동기 실행)
import asyncio


def run_async(coro):
    """Streamlit 환경에서 안전하게 비동기 함수를 실행"""
    try:
        # 기존 이벤트 루프가 있는지 확인
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 실행 중인 루프가 없으면 None
            loop = None

        if loop and loop.is_running():
            # 이미 실행 중인 루프가 있으면 nest_asyncio 사용
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            # 새 이벤트 루프 생성 및 실행
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)
    except Exception as e:
        # 이벤트 루프 생성 실패 시 최후 수단
        print(f"이벤트 루프 생성 실패: {e}")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


try:
    agent, current_tools = run_async(setup_mcp_agent())
    st.session_state["agent"] = agent
    st.session_state["tools"] = current_tools
except Exception as e:
    st.error(f"⚠️ Agent 설정 중 오류 발생: {str(e)}")
    import traceback

    st.error(f"상세 오류:\n```\n{traceback.format_exc()}\n```")
    agent = None
    current_tools = []

# 메인 채팅 인터페이스
print_messages()

# 사용자 입력
user_input = st.chat_input(
    "💬 무엇이든 물어보세요! MCP Agent가 다양한 도구를 사용하여 답변드립니다."
)

# 사용자 질문 처리
if user_input:
    if st.session_state["agent"] is None:
        st.error("⚠️ 먼저 사이드바에서 사용할 MCP 서버를 선택해주세요.")
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

                # Agent 실행
                full_response = ""
                tool_calls = []

                with st.spinner("🤔 MCP Agent가 생각하고 있습니다..."):
                    # Agent 실행하여 응답 생성 (비동기 처리)
                    async def invoke_agent():
                        return await st.session_state["agent"].ainvoke(inputs, config)

                    response = run_async(invoke_agent())

                    # 현재 턴의 메시지 추출 (역순 탐색 방식)
                    if response and "messages" in response:
                        all_messages = response["messages"]

                        # 디버깅: 전체 메시지 수 확인
                        print(f"전체 메시지 수: {len(all_messages)}")

                        # 역순으로 탐색하여 현재 턴의 메시지만 추출
                        # 마지막 human 메시지(현재 입력)부터 마지막 AI 응답까지

                        current_turn_messages = []

                        # 1단계: 마지막 human 메시지의 인덱스 찾기
                        last_human_idx = None
                        for i in range(len(all_messages) - 1, -1, -1):
                            if getattr(all_messages[i], "type", None) == "human":
                                last_human_idx = i
                                break

                        # 2단계: 마지막 human 메시지 이후의 모든 메시지가 현재 턴
                        if last_human_idx is not None:
                            current_turn_messages = all_messages[last_human_idx:]

                        # 디버깅: 현재 턴 메시지 수 확인
                        print(f"현재 턴 메시지 수: {len(current_turn_messages)}")
                        print(f"마지막 human 메시지 인덱스: {last_human_idx}")

                        # 현재 턴 메시지에서 도구 호출 정보 추출
                        for msg in current_turn_messages:
                            msg_type = getattr(msg, "type", None)

                            # 도구 호출 메시지 확인 (AI 메시지에 tool_calls 속성이 있음)
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                print(f"도구 호출 발견: {len(msg.tool_calls)}개")
                                for tool_call in msg.tool_calls:
                                    tool_info = {
                                        "name": tool_call.get("name", "Unknown Tool"),
                                        "args": tool_call.get("args", {}),
                                        "id": tool_call.get("id", "unknown"),
                                    }
                                    tool_calls.append(tool_info)

                            # 도구 실행 결과 메시지 확인
                            if msg_type == "tool":
                                tool_id = getattr(msg, "tool_call_id", None)
                                content = getattr(msg, "content", "")
                                print(f"도구 결과 발견: {tool_id}")
                                for tool_call in tool_calls:
                                    if tool_call["id"] == tool_id:
                                        tool_call["result"] = content
                                        break

                        # AI의 최종 응답 추출 (마지막 AI 메시지)
                        for msg in reversed(current_turn_messages):
                            if getattr(msg, "type", None) == "ai":
                                content = getattr(msg, "content", "")
                                if content and content.strip():
                                    full_response = content
                                    break

                # 도구 호출 정보 표시 (확장 가능한 섹션으로)
                if tool_calls:
                    with st.expander(f"🛠️ MCP 도구 호출 정보", expanded=False):
                        for i, tool_call in enumerate(tool_calls, 1):
                            st.markdown(f"**{i}. {tool_call['name']}**")

                            # 도구 호출 인자 표시
                            if tool_call["args"]:
                                st.markdown("📝 **호출 인자:**")
                                for key, value in tool_call["args"].items():
                                    # 값이 너무 긴 경우 축약
                                    if isinstance(value, str) and len(value) > 100:
                                        value = value[:100] + "..."
                                    st.markdown(f"  • `{key}`: {value}")

                            # 도구 실행 결과 표시
                            if "result" in tool_call:
                                st.markdown("📊 **실행 결과**")
                                st.markdown(tool_call["result"])

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
                st.info("💡 MCP 서버가 실행 중인지 확인하거나 다시 시도해보세요.")

# 사이드바 하단에 현재 설정 정보 표시
with st.sidebar:
    st.divider()
    st.markdown("### 📊 현재 설정")
    st.caption(f"**모델:** {selected_model}")
    st.caption(f"**Temperature:** {temperature}")
    st.caption(f"**Thread ID:** {st.session_state['thread_id'][:8]}...")
    st.caption(f"**활성 MCP 도구 개수:** {len(current_tools) if current_tools else 0}")

    # 도구별 상세 정보
    if current_tools:
        with st.expander("🔧 MCP 도구 상세 정보"):
            for i, tool in enumerate(current_tools):
                tool_name = getattr(tool, "name", f"Tool {i+1}")
                tool_desc = (
                    getattr(tool, "description", "No description available")[:100]
                    + "..."
                )
                st.caption(f"**{tool_name}:** {tool_desc}")

    # 활성 MCP 서버 정보
    st.divider()
    st.markdown("### 🌐 활성 MCP 서버")
    active_servers = []
    if use_weather_server:
        active_servers.append("🌤️ 날씨 서버 (stdio)")
    if use_time_server:
        active_servers.append("⏰ 시간 서버 (HTTP)")
    if use_rag_server:
        active_servers.append("🔍 RAG 서버 (HTTP)")
    for smithery_server in st.session_state["smithery_servers"]:
        active_servers.append(f"📦 {smithery_server['name']} (Smithery)")

    if active_servers:
        for server in active_servers:
            st.caption(f"• {server}")
    else:
        st.caption("_활성 서버 없음_")
