import os
import json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re

from agents.graph import build_agent, get_checkpointer_cm
from models.required_models import get_deployment_mode
from services.source_manager import (
    add_file_source,
    add_url_source,
    list_sources,
    remove_source,
    get_chunks_by_source
)

# app = FastAPI(title="Agentic RAG Workbench")

# Serve static frontend files
#app.mount("/static", StaticFiles(directory="static"), name="static")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    # This validates the mode on startup and crashes early if misconfigured
    mode = get_deployment_mode()
    print(f"🚀 Starting Agentic RAG Workbench in [{mode.upper()}] mode using .env configuration.")
    
    yield  # Application runs and handles requests here
    
    # --- SHUTDOWN LOGIC (Optional) ---
    print("🛑 Shutting down Agentic RAG Workbench...")


app = FastAPI(title="Agentic RAG Workbench", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


class URLRequest(BaseModel):
    url: str
    owner_id: str = "default_user"


class ChatRequest(BaseModel):
    message: str
    owner_id: str = "default_user"
    thread_id: str = "default_thread"


@app.get("/api/sources")
async def api_list_sources(owner_id: str = "default_user"):
    try:
        sources = list_sources(owner_id)
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sources/url")
async def api_add_url(data: URLRequest):
    try:
        source = add_url_source(url=data.url, owner_id=data.owner_id)
        return {"status": "success", "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sources/file")
async def api_add_file(file: UploadFile = File(...), owner_id: str = Form("default_user")):
    temp_file_path = f"/tmp/{file.filename}"
    try:
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        source = add_file_source(
            file_path=temp_file_path,
            filename=file.filename,
            owner_id=owner_id,
        )
        return {"status": "success", "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.delete("/api/sources/{source_id}")
async def api_delete_source(source_id: str, owner_id: str = "default_user"):
    try:
        success = remove_source(source_id=source_id, owner_id=owner_id)
        if not success:
            raise HTTPException(status_code=404, detail="Source not found.")
        return {"status": "success", "message": f"Source {source_id} removed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat(data: ChatRequest):
    try:
        async with get_checkpointer_cm() as checkpointer:
            agent = await build_agent(data.owner_id, checkpointer=checkpointer)
            result = await agent.ainvoke(
                {
                    "owner_id": data.owner_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": data.message,
                        }
                    ],
                    "retry_count": 0
                },
                config={"configurable": {"thread_id": data.thread_id}},
            )

        final_message = result["messages"][-1]
        return {
            "response": str(final_message.content),
            "citations": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import re

def clean_streamed_content(content: str) -> str:
    """Strips out internal model thinking tags, scratchpads, and conversational filler leaks."""
    if not content:
        return ""
    
    # 1. Remove standard reasoning tags (e.g., <think>...</think> from reasoning models)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    
    # 2. Remove unclosed or partial thinking tags if they leak during streaming
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    
    # 3. Strip leading internal monologues like 'AI: Okay, let me...' or similar patterns
    cleaned = re.sub(r"^(AI:|Okay,|Let me).*?\n", "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

def clean_streamed_content(content: str) -> str:
    """Strips out internal model thinking tags, scratchpads, and conversational filler leaks."""
    if not content:
        return ""
    
    # 1. Remove standard reasoning tags (e.g., <think>...</think> from reasoning models)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    
    # 2. Remove unclosed or partial thinking tags if they leak during streaming
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    
    # 3. Strip leading internal monologues like 'AI: Okay, let me...' or similar patterns
    cleaned = re.sub(r"^(AI:|Okay,|Let me).*?\n", "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

@app.post("/api/chat/stream")
async def api_chat_stream(payload: dict):
    message = payload.get("message")
    owner_id = payload.get("owner_id", "default_user")
    thread_id = payload.get("thread_id", "session_1")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            async with get_checkpointer_cm() as checkpointer:
                agent = await build_agent(owner_id=owner_id, checkpointer=checkpointer)
                
                async for update in agent.astream(
                    {
                        "owner_id": owner_id,
                        "messages": [{"role": "user", "content": message}],
                        "retry_count": 0
                    },
                    config=config,
                    stream_mode="updates",
                ):
                    # 1. Stream step indicators to the UI (Drives your status badges / execution timeline)
                    for node_name in update.keys():
                        step_labels = {
                            "router": "Analyzing user intent...",
                            "rewriter": "Optimizing search query...",
                            "retrieval": "Retrieving and reranking sources...",
                            "grader": "Evaluating evidence sufficiency...",
                            "refine": "Refining search terms...",
                            "generator": "Synthesizing grounded response...",
                            "grounding": "Verifying factual grounding...",
                            "abstain": "Synthesizing final fallback..."
                        }
                        if node_name in step_labels:
                            yield f"data: {json.dumps({'status': step_labels[node_name]})}\n\n"

                    # 2. Stream tokens strictly from response nodes (generator, general_chat, abstain)
                    # Note: 'grounding' is excluded here so internal verification checks don't leak text into chat
                    # Stream tokens only from the primary generator/response nodes smoothly
                    for node_name in ("generator", "general_chat", "abstain"):
                        node_update = update.get(node_name)
                        if not node_update:
                            continue
                        messages = node_update.get("messages", [])
                        if messages:
                            raw_content = getattr(messages[-1], "content", None)
                            if raw_content:
                                safe_content = clean_streamed_content(raw_content)
                                if safe_content:
                                    # Yield tokens continuously without inserting block dividers
                                    yield f"data: {json.dumps({'token': safe_content})}\n\n"

                yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/history")
async def api_get_chat_history(thread_id: str = "default_thread", owner_id: str = "default_user"):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Open checkpointer and build agent with correct owner context
        async with get_checkpointer_cm() as checkpointer:
            agent = await build_agent(owner_id=owner_id, checkpointer=checkpointer)
            
            state_snapshot = await agent.aget_state(config)
            
            if not state_snapshot or not state_snapshot.values:
                return {"messages": []}
            
            messages = state_snapshot.values.get("messages", [])
            serialized_messages = []
            
            for msg in messages:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role = "User" if msg.type in ("human", "user") else "AI"
                    serialized_messages.append({"sender": role, "text": msg.content})
                elif isinstance(msg, dict):
                    role = "User" if msg.get("role") in ("human", "user") else "AI"
                    serialized_messages.append({"sender": role, "text": msg.get("content", "")})
                    
            return {"messages": serialized_messages}
            
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return {"messages": []}


@app.get('/api/sources/{source_id}/chunks')
async def api_get_source_chunks(source_id: str, owner_id: str = "default_user"):
    try:
        chunks = get_chunks_by_source(source_id=source_id, owner_id=owner_id)
        return {"source_id": source_id, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)