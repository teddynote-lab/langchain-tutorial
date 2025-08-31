# 📚 LangChain Tutorial

> **LangChain의 모든 것을 마스터하세요**  
> 이 종합 가이드는 초급자부터 고급 개발자까지 LangChain의 핵심 개념부터 실무 활용까지 체계적으로 학습할 수 있도록 설계된 한국어 튜토리얼입니다. 실전 프로젝트와 심화 실습을 통해 강력한 AI 애플리케이션을 구축하는 전문 역량을 키워보세요.

<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://python.langchain.com/img/brand/wordmark.png">
  <source media="(prefers-color-scheme: dark)" srcset="https://python.langchain.com/img/brand/wordmark-dark.png">
  <img alt="LangChain Logo" src="https://python.langchain.com/img/brand/wordmark.png" width="80%">
</picture>

<div>
<br>
</div>

[![Version](https://img.shields.io/pypi/v/langchain.svg)](https://pypi.org/project/langchain/)
[![Downloads](https://static.pepy.tech/badge/langchain/month)](https://pepy.tech/project/langchain)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain)](https://github.com/langchain-ai/langchain/issues)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://python.langchain.com/)

**LangChain**은 Large Language Models(LLMs)을 활용한 애플리케이션 개발을 위한 포괄적인 프레임워크입니다. 모듈형 설계와 강력한 추상화를 통해 복잡한 AI 워크플로우를 단순화하고, 개발자가 빠르게 프로덕션 수준의 AI 애플리케이션을 구축할 수 있도록 지원합니다.

## 🎯 핵심 기능

- **LCEL (LangChain Expression Language)**: 체인을 구성하고 호출하는 선언적 방식으로 복잡한 워크플로우를 직관적으로 표현
- **모듈형 컴포넌트**: Models, Prompts, Output parsers, Retrievers, Memory 등 재사용 가능한 구성 요소들
- **RAG (Retrieval-Augmented Generation)**: 외부 지식 소스와 LLM을 결합하여 정확하고 최신 정보 기반 응답 생성
- **에이전트 시스템**: 도구 사용, 계획 수립, 실행을 통한 자율적 문제 해결 능력
- **메모리 관리**: 대화 히스토리 및 컨텍스트를 효율적으로 관리하는 다양한 메모리 구조
- **스트리밍 지원**: 실시간 응답 스트리밍으로 사용자 경험 향상

## ⬇️ 프로젝트 다운로드

다음 명령어를 사용하여 프로젝트를 다운로드하십시오:

```bash
git clone https://github.com/teddynote-lab/langchain-tutorial.git
cd langchain-tutorial
```

## 🔧 설치 방법

### UV 패키지 매니저를 사용한 설치

본 프로젝트는 `uv` 패키지 매니저를 사용하여 의존성을 관리합니다. 다음 단계를 따라 설치하십시오.

#### UV 설치 (사전 요구사항)

**macOS:**
```bash
# Homebrew 사용
brew install uv

# 또는 curl 사용
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
# PowerShell 사용
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip 사용
pip install uv
```

#### 프로젝트 의존성 설치

UV가 설치되었다면, 다음 명령어로 프로젝트 의존성을 설치하십시오:

```bash
uv sync
```

이 명령어는 가상 환경을 자동으로 생성하고 모든 필요한 의존성을 설치합니다.

#### 가상 환경 활성화

```bash
# 가상 환경 활성화
source .venv/bin/activate  # macOS/Linux

# 또는
.venv\Scripts\activate     # Windows
```

## 📁 프로젝트 폴더 구성

```
langchain-tutorial/
├── 01-LCEL/                # LangChain Expression Language 학습
│   ├── 01-Basic-LCEL.ipynb
│   ├── 02-Parallel-Components.ipynb
│   ├── 03-Conditional-Logic.ipynb
│   ├── 04-Streaming.ipynb
│   └── 05-Advanced-LCEL.ipynb
├── 02-RAG/                 # Retrieval-Augmented Generation
│   ├── 01-Basic-RAG.ipynb
│   ├── 02-Document-Loaders.ipynb
│   ├── 03-Text-Splitters.ipynb
│   ├── 04-Vector-Stores.ipynb
│   ├── 05-Retrievers.ipynb
│   └── 06-Advanced-RAG.ipynb
├── 03-Summary/             # 텍스트 요약 시스템
│   ├── 01-Map-Reduce-Summary.ipynb
│   ├── 02-Stuff-Summary.ipynb
│   ├── 03-Refine-Summary.ipynb
│   └── 04-Custom-Summary.ipynb
├── 04-Agent/               # 에이전트 시스템
│   ├── 01-Tools.ipynb
│   ├── 02-Bind-Tools.ipynb
│   ├── 03-Tool-Calling-Agent.ipynb
│   └── 04-React-Agent.ipynb
├── 05-MCP/                 # Model Context Protocol
│   ├── 01-MCP-Basics.ipynb
│   ├── 02-Custom-MCP.ipynb
│   └── 03-MCP-Integration.ipynb
├── 06-Modules/             # 핵심 모듈별 심화 학습
│   ├── 01-Basic/           # LangChain 기초 개념
│   ├── 02-Prompt/          # 프롬프트 템플릿과 엔지니어링
│   ├── 03-OutputParser/    # 출력 파서와 구조화
│   ├── 04-Model/           # 언어 모델 통합
│   ├── 05-Memory/          # 메모리 시스템
│   ├── 06-DocumentLoader/  # 문서 로더
│   ├── 07-TextSplitter/    # 텍스트 분할
│   ├── 08-Embeddings/      # 임베딩
│   ├── 09-VectorStore/     # 벡터 저장소
│   ├── 10-Retriever/       # 검색기
│   ├── 11-Reranker/        # 재순위화
│   └── 12-LangChain-Expression-Language/  # LCEL 심화
└── 99-Project/             # 실전 프로젝트
    ├── 01-Chatbot-Project.ipynb
    ├── 02-RAG-System.ipynb
    └── 03-Agent-Application.ipynb
```

### 폴더별 상세 설명

- **01-LCEL/**: LangChain Expression Language의 기본 사용법부터 고급 패턴까지
- **02-RAG/**: 문서 검색 및 생성 통합 시스템 구현을 위한 완전한 가이드
- **03-Summary/**: 다양한 텍스트 요약 전략과 구현 방법
- **04-Agent/**: 도구 사용과 자율적 의사결정이 가능한 에이전트 시스템 구축
- **05-MCP/**: Model Context Protocol을 활용한 고급 통합 기법
- **06-Modules/**: 각 핵심 컴포넌트별 심화 학습
  - **01-Basic/**: LangChain의 기본 개념과 구조 이해
  - **02-Prompt/**: 효과적인 프롬프트 템플릿 설계 및 엔지니어링 기법
  - **03-OutputParser/**: LLM 출력의 구조화와 파싱 기술
  - **04-Model/**: OpenAI, Anthropic 등 다양한 언어 모델 통합 방법
  - **05-Memory/**: 대화 히스토리 및 컨텍스트 메모리 관리 시스템
  - **06-DocumentLoader/**: 다양한 형식의 문서 로딩 및 전처리
  - **07-TextSplitter/**: 효율적인 텍스트 분할 전략과 구현
  - **08-Embeddings/**: 텍스트 임베딩 생성 및 활용 방법
  - **09-VectorStore/**: 벡터 데이터베이스 구축 및 관리
  - **10-Retriever/**: 정보 검색 시스템 설계 및 최적화
  - **11-Reranker/**: 검색 결과 재순위화를 통한 정확도 향상
  - **12-LangChain-Expression-Language/**: LCEL을 활용한 고급 체인 구성
- **99-Project/**: 실제 비즈니스 시나리오를 위한 완성형 프로젝트

## 🔗 참고 링크

### 📚 공식 문서 및 리포지토리
- [LangChain 공식 GitHub](https://github.com/langchain-ai/langchain) - LangChain 소스 코드 및 최신 업데이트
- [LangChain 공식 문서](https://python.langchain.com/) - 상세한 API 문서 및 가이드

### 🎓 학습 자료
- [테디노트 유튜브 채널](https://www.youtube.com/c/teddynote) - AI/ML 관련 한국어 강의 및 튜토리얼
- [RAG 고급 온라인 강의](https://fastcampus.co.kr/data_online_teddy) - 체계적인 RAG 시스템 구축 강의

## 📄 라이센스

본 프로젝트의 라이센스 정보는 [LICENSE](./LICENSE) 파일을 참조하십시오.

## 🏢 제작자

**Made by TeddyNote LAB**