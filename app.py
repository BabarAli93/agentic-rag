import os
import json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

from agents.rag_agent import build_agent, get_checkpointer_cm
from services.source_manager import (
    add_file_source,
    add_url_source,
    list_sources,
    remove_source,
    get_chunks_by_source
)


app = FastAPI(title="Agentic RAG Notebook Workbench")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        agent = build_agent(data.owner_id)
        
        # Invoke LangGraph agent
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": data.message,
                    }
                ]
            },
            config={"configurable": {"thread_id": data.thread_id}},
        )

        final_message = result["messages"][-1]
        
        # Extract citations if your graph populates them, or return content
        return {
            "response": str(final_message.content),
            "citations": []  # Hook into your retrieved source metadata chunks if available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def api_chat_stream(payload: dict):

    message = payload.get("message")
    owner_id = payload.get("owner_id", "default_user")
    thread_id = payload.get("thread_id", "session_1")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    #agent = build_agent(owner_id=owner_id)
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():

        try:

            # Safely open the request-bound SQLite checkpointer context
            # Stream with LangGraph agent
            async with get_checkpointer_cm() as checkpointer:
                agent = await build_agent(owner_id=owner_id, checkpointer=checkpointer)
                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": message}]},
                    config=config,
                    version="v2"
                    ):

                    kind = event.get("event")
                    # Stream LLM generated tokens

                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and chunk.content:
                            # Send token as SSE data packet
                            yield f"data: {json.dumps({'token': chunk.content})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/history")
async def api_get_chat_history(thread_id: str = "default_thread"):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Open checkpointer and build agent to fetch state cleanly
        async with get_checkpointer_cm() as checkpointer:
            agent = await build_agent(owner_id="default_user", checkpointer=checkpointer)
            
            # Fetch current graph state using official get_state method
            state_snapshot = await agent.aget_state(config)
            
            if not state_snapshot or not state_snapshot.values:
                return {"messages": []}
            
            messages = state_snapshot.values.get("messages", [])
            serialized_messages = []
            
            for msg in messages:
                # Handle LangChain message classes or dictionary structures
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
async def api_get_source_chunks(source_id:str, owner_id: str = "default_user"):
    try:
        chunks = get_chunks_by_source(source_id=source_id, owner_id=owner_id)
        return {"source_id": source_id, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)