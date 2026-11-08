import os
import json
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re

from agents.graph import build_agent, get_checkpointer_cm
from models.required_models import get_deployment_mode
from security.security_layer import security, AuthenticatedUser
from services.source_manager import (
    add_file_source,
    add_url_source,
    list_sources,
    remove_source,
    get_chunks_by_source,
    reindex_source
)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    security.validate_configuration()
    mode = get_deployment_mode()
    print(f"🚀 Starting Agentic RAG Workbench in [{mode.upper()}] mode using .env configuration.")
    yield
    print("🛑 Shutting down Agentic RAG Workbench...")

app = FastAPI(title="Agentic RAG Workbench", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

class URLRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"


# =====================================================================
# HELPERS
# =====================================================================
def get_scoped_thread_id(
    current_user: AuthenticatedUser,
    client_thread_id: str
) -> str:

    return (
        f"{current_user.tenant_id}:"
        f"{current_user.user_id}:"
        f"{client_thread_id}"
    )


def clean_streamed_content(content: str) -> str:
    """Strips out internal model thinking tags, scratchpads, and conversational filler leaks."""
    if not content:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^(AI:|Okay,|Let me).*?\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

# =====================================================================
# SOURCE MANAGEMENT ENDPOINTS (Secured with Tenant ID)
# =====================================================================

@app.get("/api/sources")
async def api_list_sources(
    current_user: AuthenticatedUser = Depends(
        security.require_roles("admin", "user", "read-only"))):
    tenant_id = current_user.tenant_id
    try:
        sources = list_sources(tenant_id)
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sources/url")
async def api_add_url(data: URLRequest, 
                      current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user")
    )):
    tenant_id = current_user.tenant_id
    try:
        source = add_url_source(url=data.url, owner_id=tenant_id)
        return {"status": "success", "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sources/file")
async def api_add_file(file: UploadFile = File(...), 
                       current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))):
    permanent_file_path = UPLOAD_DIR / f"{uuid4()}__{file.filename}"
    tenant_id = current_user.tenant_id
    try:
        contents = await file.read()
        with open(permanent_file_path, "wb") as f:
            f.write(contents)
        source = add_file_source(file_path=str(permanent_file_path), filename=file.filename, owner_id=tenant_id)
        return {"status": "success", "source": source}
    except Exception as e:
        if permanent_file_path.exists():
            permanent_file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sources/{source_id}")
async def api_delete_source(source_id: str, 
                            current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))):
    tenant_id = current_user.tenant_id
    try:
        success = remove_source(source_id=source_id, owner_id=tenant_id)
        if not success:
            raise HTTPException(status_code=404, detail="Source not found.")
        return {"status": "success", "message": f"Source {source_id} removed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sources/{source_id}/reindex")
async def api_reindex_source(source_id: str, 
                             current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))):
    tenant_id = current_user.tenant_id
    try:
        source = reindex_source(source_id=source_id, owner_id=tenant_id)
        return {"status": "success", "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/sources/{source_id}/chunks')
async def api_get_source_chunks(source_id: str, 
                                current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))):
    tenant_id = current_user.tenant_id
    try:
        chunks = get_chunks_by_source(source_id=source_id, owner_id=tenant_id)
        return {"source_id": source_id, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# UNIFIED SECURE CHAT ENDPOINT (Blocking JSON)
# =====================================================================
@app.post("/api/chat")
async def secure_chat_endpoint(
    req: ChatRequest,
    current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))
):
    tenant_id = current_user.tenant_id
    thread_id = get_scoped_thread_id(
        current_user,
        req.thread_id
    )

    # 1. Fail-Closed Input Guardrail Check (Asynchronous)
    safety_check = await security.validate_input_fail_closed(req.message)
    if not safety_check["safe"]:
        return {
            "response": "I'm sorry, but your request cannot be processed due to safety or security policies.",
            "citations": [],
            "abstained": True,
            "safety_violation": safety_check["category"],
            "reason": safety_check["reason"]
        }

    # 2. Execute LangGraph Agent with Verified Tenant ID (Prevents IDOR/BOLA)
    try:
        async with get_checkpointer_cm() as checkpointer:
            agent = await build_agent(tenant_id, checkpointer=checkpointer)
            result = await agent.ainvoke(
                {
                    "owner_id": tenant_id,
                    "messages": [{"role": "user", "content": req.message}],
                    "retry_count": 0
                },
                config={"configurable": {"thread_id": thread_id}},
            )

        final_message = result["messages"][-1]
        raw_response = str(final_message.content)
        citations = result.get("citations", [])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Output Guardrail Check (PII / Secret Key Leakage)
    output_safety_check = security.validate_output(raw_response)
    if not output_safety_check["safe"]:
        return {
            "response": "I'm sorry, the generated response violated security policies and was withheld.",
            "citations": [],
            "abstained": True,
            "safety_violation": output_safety_check["category"],
            "reason": output_safety_check["reason"]
        }

    return {
        "response": raw_response,
        "citations": citations,
        "abstained": False,
        "tenant_id": tenant_id
    }


# =====================================================================
# UNIFIED SECURE CHAT STREAMING ENDPOINT (SSE)
# =====================================================================
@app.post("/api/chat/stream")
async def secure_chat_stream_endpoint(
    req: ChatRequest,
    current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))
):
    tenant_id = current_user.tenant_id
    thread_id = get_scoped_thread_id(
        current_user,
        req.thread_id
    )

    # 1. Fail-Closed Input Guardrail Check (Asynchronous)
    safety_check = await security.validate_input_fail_closed(req.message)
    if not safety_check["safe"]:
        async def blocked_generator():
            yield f"data: {json.dumps({'error': 'Request blocked by security policy', 'reason': safety_check['reason']})}\n\n"
        return StreamingResponse(blocked_generator(), media_type="text/event-stream")

    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            async with get_checkpointer_cm() as checkpointer:
                agent = await build_agent(owner_id=tenant_id, checkpointer=checkpointer)

                final_citations = []
                async for update in agent.astream(
                    {
                        "owner_id": tenant_id,
                        "messages": [{"role": "user", "content": req.message}],
                        "retry_count": 0
                    },
                    config=config,
                    stream_mode="updates",
                ):
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

                    if "generator" in update and "citations" in update["generator"]:
                        final_citations = update["generator"]["citations"]

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
                                    yield f"data: {json.dumps({'token': safe_content})}\n\n"

                yield f"data: {json.dumps({'done': True, 'citations': final_citations})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# =====================================================================
# CHAT HISTORY ENDPOINT
# =====================================================================
@app.get("/api/chat/history")
async def api_get_chat_history(thread_id: str = "default_thread", 
                               current_user: AuthenticatedUser = Depends(security.require_roles("admin", "user"))):
    tenant_id = current_user.tenant_id
    scoped_thread_id = get_scoped_thread_id(
        current_user,
        thread_id
    )
    try:
        config = {"configurable": {"thread_id": scoped_thread_id}}
        async with get_checkpointer_cm() as checkpointer:
            agent = await build_agent(owner_id=tenant_id, checkpointer=checkpointer)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)