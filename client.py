"""Single-file Streamlit client. Run: python -m streamlit run simple_client.py
 
Uses main.py, manim_server.py and weather_server.py from the corrected project.
No runtime.py or server_config.py imports are needed.
"""
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
 
import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
 
# 1. Environment variables
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
HORIZON_API_KEY = os.getenv("HORIZON_API_KEY", "").strip()
 
# 2. Server declarations -- the same command / args / env / url structure.
# Local servers are separate programs; this file is their client.
SERVERS = {
    "demo-server": {
        "transport": "stdio",
        "command": (
            r"C:\Users\BJIT\Desktop\local-mcp-server"
            r"\.venv\Scripts\python.exe"
        ),
        "args": [
            r"C:\Users\BJIT\Desktop\local-mcp-server\main.py"
        ],
    },
 
        "manim-server": {
        "transport": "stdio",
        "command": (
            r"C:\Users\BJIT\Desktop\manim-mcp-server"
            r"\.venv\Scripts\python.exe"
        ),
        "args": [
            r"C:\Users\BJIT\Desktop\manim-mcp-server\src\manim_server.py"
        ],
        "env": {
            "MANIM_EXECUTABLE": (
                r"C:\Users\BJIT\Desktop\manim-mcp-server"
                r"\.venv\Scripts\manim.exe"
            ),
            "MANIM_RENDER_TIMEOUT": "45",
        },
    },
 
   
        "weather": {
        "transport": "stdio",
        "command": (
            "C:/Users/BJIT/Desktop/mcp-weather/"
            ".venv/Scripts/mcp-weather.exe"
        ),
        "args": [],
        "env": {
            "ACCUWEATHER_API_KEY": os.getenv("ACCUWEATHER_API_KEY", "")
        },
    },
    "remote": {
        "transport": "streamable-http",
        "url": os.getenv("REMOTE_MCP_URL", "https://farhan-mcp-remote-server.fastmcp.app/mcp"),
        "headers": {"Authorization": f"Bearer {HORIZON_API_KEY}"},
    },
}
 
# 3. Create local subprocess clients or an authenticated HTTP client.
def make_client(name, config):
    if config["transport"] == "stdio":
        transport = StdioTransport(
            command=config["command"], args=config["args"],
            env=config.get("env"), cwd=str(ROOT), keep_alive=False,
        )
    else:
        if name == "remote" and not HORIZON_API_KEY:
            raise ValueError("Set HORIZON_API_KEY in .env, then restart Streamlit.")
        transport = StreamableHttpTransport(
            url=config["url"],
            headers=config.get("headers"),
        )
    return Client(transport, mode="legacy", init_timeout=180,
                  timeout=75 if name.startswith("manim") else 60)
 
 
# 4. Prompt and small display helpers
SYSTEM_PROMPT = """Use connected tools for the user's request. Never invent results.
Return a concise answer after tools finish. If a tool fails, explain the error;
do not keep retrying it. For Manim, define one Scene called GeneratedScene.
Use short animations, preferably shapes or Text. MathTex requires LaTeX.
Rendering is synchronous: there is no job-status polling tool.
"""
 
 
def display_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content if isinstance(block, (str, dict))
    )
 
 
def error_text(error):
    if isinstance(error, BaseExceptionGroup):
        return " | ".join(error_text(e) for e in error.exceptions)
    text = str(error) or type(error).__name__
    for variable in ("GOOGLE_API_KEY", "ACCUWEATHER_API_KEY", "HORIZON_API_KEY"):
        if value := os.getenv(variable):
            text = text.replace(value, "[redacted]")
    return text
 
 
# 5. Connect -> load tools -> bind Gemini -> run tools -> final answer.
# Connections stay open for the WHOLE turn, then close together.
# No async clients or subprocesses are left attached to a closed event loop.
async def run_client(messages=None):
    names = list(SERVERS) if messages is None else list(st.session_state.loaded)
    st.session_state.loaded = {}
    st.session_state.errors = {}
    async with AsyncExitStack() as connections:
        tools = []
        for name in names:
            try:
                with st.spinner(f"Connecting {name}…"):
                    async with asyncio.timeout(180):
                        adapter = await connections.enter_async_context(
                            MCPAdapter(make_client(name, SERVERS[name]))
                        )
                        server_tools = await adapter.list_tools()
                for tool in server_tools:
                    tool.name = f"{name}_{tool.name}"
                tools.extend(server_tools)
                st.session_state.loaded[name] = [t.name for t in server_tools]
            except Exception as exc:
                st.session_state.errors[name] = error_text(exc)
 
        if messages is None:  # Just discover tools when the Load button is clicked.
            return
        if not tools:
            raise RuntimeError("No tools connected. Check the sidebar errors and click Load / retry servers.")
 
        tool_by_name = {tool.name: tool for tool in tools}
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0, max_retries=1,
        )
        llm_with_tools = llm.bind_tools(tools)
        try:
            for _ in range(6):
                response = await asyncio.wait_for(llm_with_tools.ainvoke(messages), 90)
                messages.append(response)  # Keep Gemini metadata and tool-call IDs.
                if not response.tool_calls:
                    return messages
 
                for call in response.tool_calls:
                    name = call["name"]
                    with st.status(f"Running {name}", expanded=True) as status:
                        st.json(call.get("args", {}))
                        try:
                            limit = 75 if name.startswith("manim") else 60
                            result = await asyncio.wait_for(
                                tool_by_name[name].ainvoke(call.get("args", {})), limit
                            )
                            content = result if isinstance(result, str) else json.dumps(result, default=str)
                            st.code(content, language="json")
                            status.update(label=f"{name}: response received", state="complete", expanded=False)
                        except Exception as exc:
                            content = f"Tool error: {error_text(exc)}"
                            st.error(content)
                            status.update(state="error")
                    messages.append(ToolMessage(
                        content=content, name=name, tool_call_id=call["id"]
                    ))
            messages.append(AIMessage(content="Stopped after six tool rounds. Check the tool results."))
            return messages
        finally:
            await llm.client.aio.aclose()
            llm.client.close()
 
 
# 6. Streamlit chat -- same message-history pattern as client2.py.
def main():
    st.set_page_config(page_title="Gemini MCP Client", page_icon="🧰")
    st.title("Gemini MCP Client")
    if "history" not in st.session_state:
        st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
        st.session_state.loaded = {}
        st.session_state.errors = {}
    if "video_since" not in st.session_state:
        # Do not display MP4 files created before this Streamlit session.
        st.session_state.video_since = time.time()
 
    with st.sidebar:
        st.caption("Load all four servers. Remote access uses HORIZON_API_KEY from .env.")
        if st.button("Load / retry servers"):
            try:
                asyncio.run(run_client())
                st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
                st.session_state.video_since = time.time()
            except Exception as exc:
                st.error(error_text(exc))
        if st.button("Clear chat"):
            st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
            st.session_state.video_since = time.time()
            st.rerun()
        server_status = st.empty()
 
    for message in st.session_state.history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(display_text(message.content))
        elif isinstance(message, AIMessage) and not message.tool_calls:
            with st.chat_message("assistant"):
                st.markdown(display_text(message.content))
 
    if prompt := st.chat_input("Ask a question…", disabled=not st.session_state.loaded):
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            # Save only a completed turn, avoiding orphan tool calls after failures.
            messages = list(st.session_state.history) + [HumanMessage(content=prompt)]
            with st.spinner("Working…"):
                history = asyncio.run(run_client(messages))
            st.session_state.history = history
            with st.chat_message("assistant"):
                st.markdown(display_text(history[-1].content))
        except Exception as exc:
            st.error(error_text(exc))
 
    with server_status.container():
        for name, names in st.session_state.loaded.items():
            with st.expander(f"{name}: {len(names)} tools available"):
                st.write(names)
        for name, error in st.session_state.errors.items():
            st.error(f"{name}: {error}")
 
    # The cloned Manim MCP server writes output beneath src/media. Show only
    # the newest complete video generated during the current chat session.
    manim_media_dir = Path(
        r"C:\Users\BJIT\Desktop\manim-mcp-server\src\media"
    )
    videos = []
    if manim_media_dir.exists():
        videos = [
            video
            for video in manim_media_dir.rglob("*.mp4")
            if "partial_movie_files" not in video.parts
            and video.stat().st_mtime >= st.session_state.video_since
        ]
    if videos:
        latest = max(videos, key=lambda video: video.stat().st_mtime)
        st.caption(f"Latest rendered animation: {latest.name}")
        st.video(str(latest))
 
 
if __name__ == "__main__":
    main()
 
 