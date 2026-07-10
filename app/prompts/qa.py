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


def select_contexts(
    contexts: list[RetrievalResult],
    max_contexts: int,
    max_context_chars: int,
) -> list[RetrievalResult]:
    """选择真正发送给大模型的上下文。

    当前策略很朴素：
    1. 默认认为 contexts 已经由 rerank 按相关性从高到低排好序。
    2. 从前往后选择，最多选择 max_contexts 个 chunk。
    3. 同时控制 chunk.text 的累计字符数不超过 max_context_chars。
    4. 如果最后一个 chunk 超过剩余字符预算，就只截取它前面的部分。

    这里不做摘要、不做语义压缩、不重新排序；只是控制发送给 LLM 的上下文预算。
    """

    if max_contexts <= 0 or max_context_chars <= 0:
        return []

    selected: list[RetrievalResult] = []
    used_chars = 0
    for result in contexts:
        if len(selected) >= max_contexts:
            break

        remaining_chars = max_context_chars - used_chars
        if remaining_chars <= 0:
            break

        text = result.chunk.text
        if len(text) > remaining_chars:
            # 最后一个可选 chunk 太长时，只裁剪 text；其他来源、权限、分数信息保持不变。
            result = result.model_copy(
                update={"chunk": result.chunk.model_copy(update={"text": text[:remaining_chars]})}
            )

        selected.append(result)
        used_chars += len(result.chunk.text)

    return selected


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
