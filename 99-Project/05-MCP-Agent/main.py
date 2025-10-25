import json
import os
import sys
import uuid
from pathlib import Path
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

# 설정 파일 관리를 위한 상수
CONFIG_DIR = Path(__file__).parent / "tool_config"
DEFAULT_CONFIG_FILE = "default-config.json"


# ===== 설정 관리 유틸리티 함수 =====
def ensure_config_directory():
    """tool_config 디렉토리가 존재하는지 확인하고 없으면 생성"""
    CONFIG_DIR.mkdir(exist_ok=True)


def save_config(filename: str, config_data: Dict[str, Any]) -> bool:
    """
    설정을 JSON 파일로 저장합니다.

    Args:
        filename: 저장할 파일명 (.json 확장자 포함 또는 미포함)
        config_data: 저장할 설정 데이터

    Returns:
        bool: 저장 성공 여부
    """
    try:
        ensure_config_directory()

        # .json 확장자가 없으면 추가
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        filepath = CONFIG_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"❌ 설정 저장 실패: {e}")
        return False


def load_config(filename: str) -> Dict[str, Any] | None:
    """
    JSON 파일에서 설정을 로드합니다.

    Args:
        filename: 로드할 파일명

    Returns:
        Dict: 로드된 설정 데이터, 실패 시 None
    """
    try:
        filepath = CONFIG_DIR / filename

        if not filepath.exists():
            print(f"⚠️ 설정 파일을 찾을 수 없습니다: {filename}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return config_data
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return None


def get_config_files() -> List[str]:
    """
    tool_config 디렉토리의 모든 JSON 설정 파일 목록을 반환합니다.

    Returns:
        List[str]: 설정 파일명 리스트
    """
    try:
        ensure_config_directory()
        config_files = [f.name for f in CONFIG_DIR.glob("*.json")]
        return sorted(config_files)
    except Exception as e:
        print(f"❌ 설정 파일 목록 조회 실패: {e}")
        return []


def apply_config_to_session(config_data: Dict[str, Any]):
    """
    로드된 설정을 세션 상태에 적용합니다.

    Args:
        config_data: 적용할 설정 데이터
    """
    if not config_data:
        return

    # MCP 서버 설정 적용
    if "mcp_servers" in config_data:
        st.session_state["mcp_servers"] = config_data["mcp_servers"]

    # Smithery 서버 설정 적용 (API 키는 빈 문자열로)
    if "smithery_servers" in config_data:
        smithery_servers = config_data["smithery_servers"]
        # API 키가 저장되어 있지 않으므로 None 또는 빈 문자열로 설정
        for server in smithery_servers:
            if "key" not in server or not server["key"]:
                server["key"] = None
        st.session_state["smithery_servers"] = smithery_servers

    # 모델 설정은 세션에 저장하지 않고 UI에서 직접 사용
    # (selected_model과 temperature는 UI 컴포넌트의 기본값으로 사용됨)

    # 커스텀 프롬프트 적용
    if "custom_prompt" in config_data:
        st.session_state["custom_prompt"] = config_data["custom_prompt"]


def get_current_config() -> Dict[str, Any]:
    """
    현재 세션 상태에서 설정을 추출합니다.

    Returns:
        Dict: 현재 설정 데이터 (API 키 제외)
    """
    # Smithery 서버 설정에서 API 키 제외
    smithery_servers_without_keys = []
    for server in st.session_state.get("smithery_servers", []):
        server_copy = server.copy()
        # API 키는 저장하지 않음 (보안상 이유)
        server_copy["key"] = None
        smithery_servers_without_keys.append(server_copy)

    return {
        "mcp_servers": st.session_state.get("mcp_servers", []),
        "smithery_servers": smithery_servers_without_keys,
        "selected_model": st.session_state.get("selected_model", "openai/gpt-4.1"),
        "temperature": st.session_state.get("temperature", 0.1),
        "custom_prompt": st.session_state.get("custom_prompt", ""),
    }


# Streamlit 앱 제목 설정
st.title("MCP Agent 챗봇")

# 세션 상태 초기화 - default-config.json에서 로드
if "config_loaded" not in st.session_state:
    # 기본 설정 로드
    default_config = load_config(DEFAULT_CONFIG_FILE)

    if default_config:
        print(f"✅ {DEFAULT_CONFIG_FILE} 로드 완료")
        apply_config_to_session(default_config)

        # 모델 설정도 세션에 저장 (UI에서 기본값으로 사용)
        st.session_state["selected_model"] = default_config.get(
            "selected_model", "openai/gpt-4.1"
        )
        st.session_state["temperature"] = default_config.get("temperature", 0.1)
    else:
        print("⚠️ 기본 설정을 로드할 수 없습니다. 하드코딩된 기본값을 사용합니다.")
        # 기본값 설정
        st.session_state["mcp_servers"] = [
            {
                "name": "weather",
                "type": "stdio",
                "command": "uv",
                "args": ["run", "python", "../../05-MCP/server/mcp_server_local.py"],
            },
        ]
        st.session_state["smithery_servers"] = []
        st.session_state["selected_model"] = "openai/gpt-4.1"
        st.session_state["temperature"] = 0.1
        st.session_state["custom_prompt"] = (
            "당신은 스마트 MCP Agent입니다. 주어진 MCP 도구를 활용하여 사용자의 질문에 응답하세요.\n"
            "문제를 해결하기 위해 다양한 MCP 도구를 사용할 수 있습니다.\n"
            "답변은 친근감 있는 어조로 답변하세요."
        )

    # 설정 로드 완료 플래그
    st.session_state["config_loaded"] = True

# 기타 세션 상태 초기화
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
if "current_config_file" not in st.session_state:
    # 현재 로드된 설정 파일명 (표시용)
    st.session_state["current_config_file"] = DEFAULT_CONFIG_FILE


async def setup_mcp_client(server_configs: Dict[str, Any]):
    """
    MCP 클라이언트를 설정하고 도구를 가져옵니다.
    실패한 서버는 건너뛰고 성공한 서버만으로 진행합니다.

    Args:
        server_configs: 서버 설정 정보 딕셔너리

    Returns:
        tuple: (MCP 클라이언트, 도구 리스트, 실패한 서버 정보 딕셔너리)
    """
    failed_servers = {}  # {서버명: 에러 메시지}
    successful_configs = {}
    all_tools = []

    print(f"\n🔄 {len(server_configs)}개의 MCP 서버 연결 시도 중...")

    # 각 서버별로 개별 연결 시도
    for server_name, config in server_configs.items():
        try:
            print(f"  📡 {server_name} 서버 연결 중...")

            # 단일 서버 클라이언트 생성
            single_server_config = {server_name: config}
            client = MultiServerMCPClient(single_server_config)

            # 도구 로드 시도
            tools = await client.get_tools()

            # 성공한 경우
            if tools:
                all_tools.extend(tools)
                successful_configs[server_name] = config
                print(f"  ✅ {server_name}: {len(tools)}개 도구 로드 성공")
            else:
                print(f"  ⚠️ {server_name}: 도구를 찾을 수 없습니다")
                failed_servers[server_name] = "도구를 찾을 수 없습니다"

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ {server_name}: 연결 실패 - {error_msg[:100]}")

            # 에러 타입별 친절한 메시지 생성
            if "Connection closed" in error_msg or "McpError" in error_msg:
                friendly_msg = (
                    "서버 프로세스가 즉시 종료되었습니다. 명령어와 경로를 확인하세요."
                )
            elif "401" in error_msg or "Unauthorized" in error_msg:
                friendly_msg = "인증 실패. API 키를 확인하세요."
            elif "ConnectError" in error_msg or "connection" in error_msg.lower():
                friendly_msg = (
                    "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
                )
            elif (
                "command not found" in error_msg.lower() or "No such file" in error_msg
            ):
                friendly_msg = "명령어를 찾을 수 없습니다. 설치 상태를 확인하세요."
            else:
                friendly_msg = error_msg[:100]

            failed_servers[server_name] = friendly_msg

    # 결과 요약
    print(f"\n📊 MCP 서버 연결 결과:")
    print(f"  ✅ 성공: {len(successful_configs)}개")
    print(f"  ❌ 실패: {len(failed_servers)}개")

    if all_tools:
        print(f"\n✅ 총 {len(all_tools)}개의 MCP 도구가 로드되었습니다:")
        for tool in all_tools:
            print(f"  - {tool.name}")

        # 성공한 서버들로 최종 클라이언트 생성
        final_client = (
            MultiServerMCPClient(successful_configs) if successful_configs else None
        )
        return final_client, all_tools, failed_servers
    else:
        print("⚠️ 연결된 MCP 서버가 없습니다.")
        return None, [], failed_servers


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
        tuple: (LangGraph 에이전트, 도구 리스트, 실패한 서버 정보)
    """
    # MCP 클라이언트 생성 및 모든 서버의 도구를 통합 로딩
    client, tools, failed_servers = await setup_mcp_client(
        server_configs=server_configs
    )

    # OpenRouter를 통한 LLM 설정
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )

    # 기존 메모리 인스턴스 사용 (대화 기록 유지)
    memory = st.session_state["memory"]

    # 도구가 없으면 Agent를 생성할 수 없음
    if not tools:
        print("⚠️ 사용 가능한 도구가 없어 Agent를 생성할 수 없습니다.")
        return None, [], failed_servers

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

    return agent, tools, failed_servers


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

    # 설정 저장/불러오기
    st.subheader("💾 설정 관리")

    # 현재 로드된 설정 파일 표시
    st.caption(f"📂 현재 설정: `{st.session_state.get('current_config_file', 'N/A')}`")

    # 설정 불러오기
    with st.expander("📂 설정 불러오기", expanded=False):
        config_files = get_config_files()

        if config_files:
            selected_config = st.selectbox(
                "설정 파일 선택",
                config_files,
                help="불러올 설정 파일을 선택하세요.",
            )

            if st.button("✅ 설정 불러오기"):
                loaded_config = load_config(selected_config)
                if loaded_config:
                    apply_config_to_session(loaded_config)

                    # 모델 설정도 세션에 업데이트
                    st.session_state["selected_model"] = loaded_config.get(
                        "selected_model", "openai/gpt-4.1"
                    )
                    st.session_state["temperature"] = loaded_config.get(
                        "temperature", 0.1
                    )
                    st.session_state["current_config_file"] = selected_config

                    st.success(f"✅ '{selected_config}' 설정을 불러왔습니다!")
                    st.info("⚠️ 모델 설정은 페이지를 새로고침하면 UI에 반영됩니다.")
                    # Agent를 재생성하기 위해 설정 초기화
                    st.session_state["current_mcp_config"] = None
                    st.rerun()
                else:
                    st.error(f"❌ '{selected_config}' 파일을 불러올 수 없습니다.")
        else:
            st.info("📭 저장된 설정 파일이 없습니다.")

    # 설정 저장
    with st.expander("💾 설정 저장", expanded=False):
        st.info(
            "💡 **참고:**\n"
            "- API 키는 보안상 저장되지 않습니다.\n"
            "- 설정 파일은 `tool_config/` 폴더에 저장됩니다."
        )

        save_filename = st.text_input(
            "파일명 입력",
            placeholder="예: my-config",
            help="파일명을 입력하세요 (.json 확장자는 자동으로 추가됩니다).",
        )

        if st.button("💾 저장"):
            if save_filename:
                # 현재 설정 수집 (세션에서 최신 값 사용)
                current_config = get_current_config()

                # 저장
                if save_config(save_filename, current_config):
                    st.success(f"✅ 설정이 '{save_filename}.json'으로 저장되었습니다!")
                    st.session_state["current_config_file"] = (
                        f"{save_filename}.json"
                        if not save_filename.endswith(".json")
                        else save_filename
                    )
                else:
                    st.error("❌ 설정 저장에 실패했습니다.")
            else:
                st.warning("⚠️ 파일명을 입력해주세요.")

    st.divider()

    # 모델 설정
    st.subheader("✅ 모델 설정")

    # 세션에 저장된 값을 사용하여 UI 초기화
    model_options = ["openai/gpt-4.1"]
    current_model = st.session_state.get("selected_model", "openai/gpt-4.1")

    # 현재 모델이 옵션에 없으면 추가 (커스텀 모델 지원)
    if current_model not in model_options:
        model_options.insert(0, current_model)

    selected_model = st.selectbox(
        "LLM 모델 선택",
        model_options,
        index=model_options.index(current_model),
        help="사용할 LLM 모델을 선택하세요.",
    )

    # 세션에 저장
    st.session_state["selected_model"] = selected_model

    temperature = st.slider(
        "🌡️ Temperature (창의성)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("temperature", 0.1),
        step=0.1,
        help="0에 가까울수록 정확하고 일관된 답변, 1에 가까울수록 창의적인 답변",
    )

    # 세션에 저장
    st.session_state["temperature"] = temperature

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

    st.markdown("**등록된 MCP 서버:**")

    # 등록된 서버 목록 표시
    if st.session_state["mcp_servers"]:
        for idx, server in enumerate(st.session_state["mcp_servers"]):
            col1, col2 = st.columns([3, 1])
            with col1:
                # 서버 타입에 따라 아이콘 표시
                icon = "📡" if server["type"] == "stdio" else "🌐"
                st.caption(f"{icon} {server['name']} ({server['type']})")
            with col2:
                if st.button("❌", key=f"remove_mcp_{idx}"):
                    st.session_state["mcp_servers"].pop(idx)
                    st.rerun()
    else:
        st.caption("_등록된 서버가 없습니다._")

    # 새 MCP 서버 추가
    with st.expander("➕ 새 MCP 서버 추가", expanded=False):
        st.info(
            "💡 **도움말:**\n"
            "- **stdio**: 로컬 프로세스로 실행 (예: Python, Node.js 스크립트)\n"
            "- **http**: 원격 HTTP 서버 (서버가 미리 실행 중이어야 함)"
        )

        # 서버 타입 선택
        server_type = st.radio(
            "서버 타입",
            ["stdio", "http"],
            horizontal=True,
            help="MCP 서버 연결 방식을 선택하세요.",
        )

        new_server_name = st.text_input(
            "서버 이름",
            placeholder="예: my-server",
            help="MCP 설정에서 사용할 서버 식별 이름",
        )

        if server_type == "stdio":
            new_server_command = st.text_input(
                "Command",
                placeholder="예: uv, python, node",
                help="실행할 명령어",
            )
            new_server_args = st.text_area(
                "Arguments (한 줄에 하나씩)",
                placeholder="예:\nrun\npython\nserver.py",
                help="명령어 인자를 한 줄에 하나씩 입력하세요.",
                height=100,
            )

            if st.button("✅ stdio 서버 추가"):
                if new_server_name and new_server_command:
                    # args를 줄바꿈으로 분리하여 리스트로 변환
                    args_list = [
                        arg.strip()
                        for arg in new_server_args.split("\n")
                        if arg.strip()
                    ]
                    st.session_state["mcp_servers"].append(
                        {
                            "name": new_server_name,
                            "type": "stdio",
                            "command": new_server_command,
                            "args": args_list,
                        }
                    )
                    st.success(f"✅ {new_server_name} 서버가 추가되었습니다!")
                    st.rerun()
                else:
                    st.error("⚠️ 서버 이름과 Command를 입력해주세요.")

        else:  # http
            new_server_url = st.text_input(
                "서버 URL",
                placeholder="예: http://127.0.0.1:8000/mcp",
                help="HTTP MCP 서버의 엔드포인트 URL",
            )

            if st.button("✅ http 서버 추가"):
                if new_server_name and new_server_url:
                    st.session_state["mcp_servers"].append(
                        {
                            "name": new_server_name,
                            "type": "http",
                            "url": new_server_url,
                        }
                    )
                    st.success(f"✅ {new_server_name} 서버가 추가되었습니다!")
                    st.rerun()
                else:
                    st.error("⚠️ 서버 이름과 URL을 입력해주세요.")

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
        "mcp_servers": [
            (s["name"], s["type"], s.get("url", ""), s.get("command", ""))
            for s in st.session_state["mcp_servers"]
        ],
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
        print("🔄 Agent 설정이 변경되었습니다. Agent를 재생성합니다...")

        server_configs = {}

        # MCP 서버들 추가 (stdio/http)
        for mcp_server in st.session_state["mcp_servers"]:
            if mcp_server["type"] == "stdio":
                server_configs[mcp_server["name"]] = {
                    "command": mcp_server["command"],
                    "args": mcp_server["args"],
                    "transport": "stdio",
                }
            else:  # http
                server_configs[mcp_server["name"]] = {
                    "url": mcp_server["url"],
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
            try:
                agent, tools, failed_servers = await create_mcp_react_agent(
                    server_configs=server_configs,
                    model_name=selected_model,
                    temperature=temperature,
                    custom_prompt=custom_prompt,
                )
                # 설정 업데이트
                st.session_state["current_mcp_config"] = config_str
                print(
                    f"✅ Agent 생성 완료: {len(tools) if tools else 0}개의 도구 로드됨"
                )
                return agent, tools, failed_servers
            except Exception as e:
                print(f"❌ Agent 생성 실패: {e}")
                import traceback

                traceback.print_exc()
                raise
        else:
            st.session_state["current_mcp_config"] = config_str
            return None, [], {}

    # 설정이 변경되지 않았으면 기존 Agent 사용
    else:
        print("♻️ 기존 Agent 재사용")
        return (
            st.session_state["agent"],
            st.session_state["tools"],
            st.session_state.get("failed_servers", {}),
        )


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
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            except ImportError:
                print("⚠️ nest_asyncio가 설치되지 않았습니다.")
                print("pip install nest-asyncio 명령으로 설치해주세요.")
                raise
        else:
            # 새 이벤트 루프 생성 및 실행
            return asyncio.run(coro)
    except Exception as e:
        # 이벤트 루프 생성 실패 시 최후 수단
        print(f"⚠️ 이벤트 루프 실행 실패: {e}")
        import traceback

        traceback.print_exc()
        raise


try:
    agent, current_tools, failed_servers = run_async(setup_mcp_agent())
    st.session_state["agent"] = agent
    st.session_state["tools"] = current_tools
    st.session_state["failed_servers"] = failed_servers

    # 실패한 서버가 있으면 경고 메시지 표시
    if failed_servers:
        st.warning(f"⚠️ {len(failed_servers)}개의 MCP 서버 연결에 실패했습니다.")

        with st.expander("❌ 연결 실패한 서버 상세 정보", expanded=True):
            for server_name, error_msg in failed_servers.items():
                st.error(f"**{server_name}**")
                st.caption(f"📋 오류: {error_msg}")

                # 서버 타입별 해결 방법 안내
                if (
                    "서버 프로세스가 즉시 종료" in error_msg
                    or "Connection closed" in error_msg
                ):
                    st.info(
                        "**💡 해결 방법:**\n"
                        "1. 명령어 경로가 올바른지 확인하세요 (예: `python` → `python3`)\n"
                        "2. 스크립트 파일 경로가 올바른지 확인하세요\n"
                        "3. 필요한 패키지가 모두 설치되어 있는지 확인하세요\n"
                        "4. 스크립트를 직접 실행해보고 오류가 있는지 확인하세요"
                    )
                elif "인증 실패" in error_msg:
                    st.info(
                        "**💡 해결 방법:**\n"
                        "1. API 키가 올바른지 확인하세요\n"
                        "2. API 키 권한을 확인하세요\n"
                        "3. 서버가 유료 구독을 요구하는지 확인하세요"
                    )
                elif "명령어를 찾을 수 없습니다" in error_msg:
                    st.info(
                        "**💡 해결 방법:**\n"
                        "1. 명령어가 설치되어 있는지 확인하세요 (예: `uv`, `npx`, `node`)\n"
                        "2. PATH 환경 변수가 올바르게 설정되어 있는지 확인하세요\n"
                        "3. 터미널에서 명령어를 직접 실행해보세요"
                    )
                elif "서버에 연결할 수 없습니다" in error_msg:
                    st.info(
                        "**💡 해결 방법:**\n"
                        "1. HTTP 서버가 실행 중인지 확인하세요\n"
                        "2. 서버 URL이 올바른지 확인하세요\n"
                        "3. 방화벽이나 네트워크 설정을 확인하세요"
                    )

                st.divider()

            st.caption(
                "💡 사이드바에서 문제가 있는 서버를 제거하거나 설정을 수정하세요."
            )

    # 모든 서버가 실패한 경우
    if not current_tools and failed_servers:
        st.error("❌ 모든 MCP 서버 연결에 실패했습니다.")
        st.info(
            "**다음 조치를 취해주세요:**\n"
            "1. 위의 오류 메시지를 확인하고 문제를 해결하세요\n"
            "2. 사이드바에서 새로운 서버를 추가하세요\n"
            "3. 페이지를 새로고침하세요"
        )

except Exception as e:
    error_msg = str(e)

    st.error("❌ MCP Agent 초기화 중 예상치 못한 오류가 발생했습니다.")

    with st.expander("🔍 오류 상세 정보", expanded=False):
        st.code(error_msg)

        # 에러 타입별 안내
        if "401" in error_msg or "Unauthorized" in error_msg:
            st.warning(
                "**💡 인증 오류:**\n"
                "- Smithery 서버의 경우 API Key가 필요할 수 있습니다.\n"
                "- 사이드바에서 해당 서버를 제거하거나 올바른 API Key를 입력하세요."
            )
        elif "ConnectError" in error_msg or "connection" in error_msg.lower():
            st.warning(
                "**💡 연결 오류:**\n"
                "- HTTP MCP 서버가 실행 중인지 확인하세요.\n"
                "- 사이드바에서 연결할 수 없는 HTTP 서버를 삭제하세요."
            )
        elif "Connection closed" in error_msg:
            st.warning(
                "**💡 서버 종료 오류:**\n"
                "- MCP 서버 프로세스가 즉시 종료되었습니다.\n"
                "- 명령어와 파일 경로를 확인하세요.\n"
                "- 터미널에서 직접 실행해보고 오류를 확인하세요."
            )

    agent = None
    current_tools = []
    st.session_state["failed_servers"] = {}

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
                    # 매번 새로운 코루틴을 생성하여 재사용 문제 방지
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

    # MCP 서버들
    for mcp_server in st.session_state["mcp_servers"]:
        icon = "📡" if mcp_server["type"] == "stdio" else "🌐"
        active_servers.append(f"{icon} {mcp_server['name']} ({mcp_server['type']})")

    # Smithery 서버들
    for smithery_server in st.session_state["smithery_servers"]:
        active_servers.append(f"📦 {smithery_server['name']} (Smithery)")

    if active_servers:
        for server in active_servers:
            st.caption(f"• {server}")
    else:
        st.caption("_활성 서버 없음_")
