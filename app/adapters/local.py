"""不依赖外部服务的本地兜底适配器。"""

import re

from app.domain import AnswerTrace, Citation, QAAnswer, RetrievalResult
from app.ports.contracts import ChatModel, Reranker

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """轻量中英文分词，仅用于本地 rerank。"""

    tokens: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        tokens.append(lowered)
        cjk_chars = CJK_RE.findall(lowered)
        tokens.extend(cjk_chars)
        tokens.extend("".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1))
    return tokens


class SimpleReranker(Reranker):
    """本地关键词 rerank；后续可替换为真实 reranker API。"""

    async def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        query_tokens = set(tokenize(query))
        for result in results:
            overlap = len(query_tokens.intersection(tokenize(result.chunk.text)))
            result.rerank_score = result.score + overlap * 0.05
        return sorted(results, key=lambda item: item.rerank_score or item.score, reverse=True)


class GroundedStubChatModel(ChatModel):
    """未配置真实 LLM 时的安全兜底模型。"""

    name = "grounded-stub-model"

    async def answer(self, query: str, contexts: list[RetrievalResult]) -> QAAnswer:
        if not contexts:
            return QAAnswer(
                answer="未在当前用户有权限访问的知识库中找到可靠依据，暂不回答。",
                citations=[],
                confidence=0.0,
                trace=AnswerTrace(
                    query=query,
                    rewritten_query=query,
                    user_subjects=set(),
                    retrieval_count=0,
                    reranked_count=0,
                    used_model=self.name,
                    refusal_reason="no_authorized_context",
                ),
            )

        top_contexts = contexts[:3]
        citations = [
            Citation(
                document_id=result.chunk.document_id,
                chunk_id=result.chunk.id,
                title=result.chunk.title,
                source_uri=result.chunk.source_uri,
                page=result.chunk.page,
                quote=result.chunk.text[:180],
            )
            for result in top_contexts
        ]
        return QAAnswer(
            answer="根据已授权知识库内容：\n"
            + "\n".join(f"- {result.chunk.text[:220]}" for result in top_contexts),
            citations=citations,
            confidence=min(0.95, max(result.score for result in top_contexts)),
            trace=AnswerTrace(
                query=query,
                rewritten_query=query,
                user_subjects=set(),
                retrieval_count=len(contexts),
                reranked_count=len(top_contexts),
                used_model=self.name,
                retrieved_chunk_ids=[result.chunk.id for result in contexts],
            ),
        )
