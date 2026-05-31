from abc import ABC, abstractmethod
from typing import List, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


class BaseAgent(ABC):

    name: str        = "base"
    description: str = "General-purpose agent"

    def __init__(self, llm: BaseChatModel):
        self.llm   = llm
        self._agent = self._build_agent()

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    def get_tools(self) -> List[BaseTool]:
        return []

    def _build_agent(self):
        tools = self.get_tools()
        if tools:
            return create_react_agent(
                model=self.llm,
                tools=tools,
                prompt=self.get_system_prompt(),
            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human",  "{subtask}"),
        ])
        return prompt | self.llm | StrOutputParser()

    async def run(self, subtask: str, messages: list = None) -> str:
        """
        Non-streaming invoke — used internally by dispatch_node.
        Returns the complete answer as a plain string.
        """
        if self.get_tools():
            input_messages = list(messages or []) + [HumanMessage(content=subtask)] 
            state = await self._agent.ainvoke({
                "messages": input_messages
                })  
            ai = [m for m in state["messages"] if m.__class__.__name__ == "AIMessage"]
            return ai[-1].content if ai else "[No response]"
        return await self._agent.ainvoke({"subtask": subtask})

    def get_runnable(self):
        """
        Return the underlying runnable for use with astream_events().
        Called by the streaming graph nodes so astream_events() can attach
        its callback hooks and emit on_chat_model_stream events.
        """
        return self._agent

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tools={[t.name for t in self.get_tools()]})"
