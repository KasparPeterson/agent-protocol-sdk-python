from agent_protocol import Agent, Task, Step
from agent_protocol.models import AgentInfo
from agent_protocol.models import Authorization


async def task_handler(task: Task) -> None:
    print(f"task: {task.input}")
    await Agent.db.create_step(task.task_id, task.input)


async def step_handler(step: Step) -> Step:
    print(f"step: {step.input}")
    await Agent.db.create_step(step.task_id, f"Next step from step {step.name}")
    step.output = step.input
    return step


agent_info = AgentInfo(
    name="My Agent",
    description="My Agent is a Cool Agent",
    version="1.0.1",
    protocol_version="23",
)

"""config_options={
    "debug": {
        "type": "boolean",
        "default": False,
        "description": "Whether to run the agent in debug mode."
    },
}"""

authorization = Authorization(
    authorization_type="bearer_token",
    access_token="mySecretKey",
)

# Agent.setup_agent(task_handler, step_handler).start()
Agent.setup_agent(task_handler, step_handler, agent_info, authorization).start()
