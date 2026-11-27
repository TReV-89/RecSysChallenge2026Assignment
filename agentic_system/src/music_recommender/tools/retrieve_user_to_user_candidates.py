"""User-to-user (CF-BPR neighbor pooling) retrieval tool, scoped to one turn."""

from langchain_core.tools import tool

from ._common import as_tool_content, summarize_candidates


def make_retrieve_user_to_user_tool(retriever, request, collected: list):
    @tool
    def retrieve_user_to_user_candidates() -> str:
        """Find tracks supported by similar listeners' eligible training feedback.

        Only use when the listener context below gives a listener ID with a
        CF-BPR vector. Takes no arguments.

        Returns:
            Catalog tracks supported by similar listeners' positive feedback,
            each with track_id, title, artists, tags; may be empty. Predicted
            affinity from a synthetic conversation, never a confirmed like.
        """
        result = retriever.retrieve_cf(request)
        collected.append(result)
        return as_tool_content(summarize_candidates(result))

    return retrieve_user_to_user_candidates
