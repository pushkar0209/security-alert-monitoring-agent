import os
import logging
import datetime
import random

import google.cloud.logging
from google.cloud import datastore
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from mcp.server.fastmcp import FastMCP
from google.adk import Agent
from google.adk.agents import SequentialAgent

# ================= 1. ENV =================
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "hack-492515")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ================= 2. LOGGING =================
try:
    google.cloud.logging.Client(project=PROJECT_ID).setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

# ================= 3. DATABASE =================
db = datastore.Client(project=PROJECT_ID)
mcp = FastMCP("WorkspaceTools")

# ================= 4. TASK TOOLS =================

@mcp.tool()
def add_task(title: str) -> str:
    task = datastore.Entity(key=db.key('Task'))
    task.update({
        'title': title,
        'completed': False,
        'created_at': datetime.datetime.utcnow()
    })
    db.put(task)
    return f"Task '{title}' added (ID: {task.key.id})"


@mcp.tool()
def list_tasks() -> str:
    tasks = list(db.query(kind='Task').fetch())
    if not tasks:
        return "No tasks found."

    return "\n".join([
        f"{'✅' if t.get('completed') else '⏳'} {t['title']} (ID: {t.key.id})"
        for t in tasks
    ])


@mcp.tool()
def complete_task(task_id: str) -> str:
    numeric = ''.join(filter(str.isdigit, task_id))
    if not numeric:
        return "Invalid task ID"

    task = db.get(db.key('Task', int(numeric)))
    if not task:
        return "Task not found"

    task['completed'] = True
    db.put(task)
    return f"Task {numeric} completed"


# ================= 5. SYSTEM MONITORING =================

@mcp.tool()
def collect_metrics() -> dict:
    return {
        "cpu": random.randint(10, 95),
        "memory": random.randint(20, 90),
        "disk": random.randint(30, 95),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


@mcp.tool()
def detect_anomalies(metrics: dict) -> str:
    alerts = []
    if metrics.get("cpu", 0) > 85:
        alerts.append("High CPU")
    if metrics.get("memory", 0) > 80:
        alerts.append("High Memory")
    if metrics.get("disk", 0) > 90:
        alerts.append("High Disk")
    return ", ".join(alerts) if alerts else "No issues"


# ================= 6. SECURITY MONITORING =================

@mcp.tool()
def collect_security_events() -> dict:
    return {
        "failed_logins": random.randint(0, 10),
        "ip_requests": random.randint(50, 500),
        "suspicious_activity": random.choice([True, False]),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


@mcp.tool()
def detect_security_threats(events: dict) -> str:
    threats = []

    if events.get("failed_logins", 0) > 5:
        threats.append("Brute Force Attack")

    if events.get("ip_requests", 0) > 400:
        threats.append("DDoS Suspicion")

    if events.get("suspicious_activity"):
        threats.append("Suspicious Behavior Detected")

    return ", ".join(threats) if threats else "No threats"


@mcp.tool()
def store_security_alert(message: str) -> str:
    entity = datastore.Entity(key=db.key("SecurityAlerts"))
    entity.update({
        "message": message,
        "timestamp": datetime.datetime.utcnow()
    })
    db.put(entity)
    return "Security alert stored"


# ================= 7. NOTIFICATIONS =================

@mcp.tool()
def send_notification(message: str) -> str:
    entity = datastore.Entity(key=db.key("Notifications"))
    entity.update({
        "message": message,
        "timestamp": datetime.datetime.utcnow()
    })
    db.put(entity)
    return f"Notification sent: {message}"


@mcp.tool()
def store_alert(message: str) -> str:
    entity = datastore.Entity(key=db.key("Alerts"))
    entity.update({
        "message": message,
        "timestamp": datetime.datetime.utcnow()
    })
    db.put(entity)
    return "Alert stored"


# ================= 8. AGENT INSTRUCTIONS =================

def root_instruction(ctx):
    return "Process request using workflow agent"


def task_instruction(ctx):
    return f"Handle tasks: {ctx.state.get('user_input', '')}"


def monitoring_instruction(ctx):
    return """
Run system monitoring:
1. collect_metrics
2. detect_anomalies
3. If issues → store_alert + send_notification
"""


def security_instruction(ctx):
    return """
Run security monitoring:
1. collect_security_events
2. detect_security_threats
3. If threats → store_security_alert + send_notification
"""


def notification_instruction(ctx):
    return f"Send notification: {ctx.state.get('user_input', '')}"


# ================= 9. AGENTS =================

task_agent = Agent(
    name="task_agent",
    model=MODEL,
    instruction=task_instruction,
    tools=[add_task, list_tasks, complete_task]
)

monitoring_agent = Agent(
    name="monitoring_agent",
    model=MODEL,
    instruction=monitoring_instruction,
    tools=[collect_metrics, detect_anomalies, store_alert, send_notification]
)

security_agent = Agent(
    name="security_agent",
    model=MODEL,
    instruction=security_instruction,
    tools=[collect_security_events, detect_security_threats, store_security_alert, send_notification]
)

notification_agent = Agent(
    name="notification_agent",
    model=MODEL,
    instruction=notification_instruction,
    tools=[send_notification]
)

workflow_agent = SequentialAgent(
    name="workflow",
    sub_agents=[
        task_agent,
        monitoring_agent,
        security_agent,
        notification_agent
    ]
)

root_agent = Agent(
    name="root",
    model=MODEL,
    instruction=root_instruction,
    sub_agents=[workflow_agent]
)

# ================= 10. API =================

app = FastAPI()

class UserRequest(BaseModel):
    prompt: str


@app.post("/api/v1/chat")
async def chat(request: UserRequest):
    try:
        final_response = ""

        async for event in root_agent.run_async({"user_input": request.prompt}):
            if getattr(event, "text", None):
                final_response = event.text

        return {
            "status": "success",
            "response": final_response or "Processed successfully"
        }

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# ================= 11. MAIN =================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)