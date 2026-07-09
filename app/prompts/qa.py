from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.domain import RetrievalResult

SYSTEM_PROMPT = (
    "你是企业知识库问答助手。"
    "只能依据用户有权限访问的上下文回答。"
    "如果上下文不足以回答，就明确说未找到可靠依据。"
    "回答要简洁，并尽量指出依据来自哪些上下文编号。"
)


def build_context_text(contexts: list[RetrievalResult]) -> str:
    """把检索结果转换成带编号的上下文文本。"""

    return "\n\n".join(
        f"[{index}] 来源：{result.chunk.title}\n{result.chunk.text}"
        for index, result in enumerate(contexts, start=1)
    )


def build_qa_messages(query: str, contexts: list[RetrievalResult]) -> list[BaseMessage]:
    """构造发给大模型的问答 messages。"""

    context_text = build_context_text(contexts)
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"问题：{query}\n\n"
                f"已授权上下文：\n{context_text}\n\n"
                "请基于以上上下文回答。"
            )
        ),
    ]
