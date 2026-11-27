"""Direct user-to-item (CF-BPR factor) retrieval tool, scoped to one turn."""

from langchain_core.tools import tool

from ._common import as_tool_content, summarize_candidates


def make_retrieve_user_to_item_tool(retriever, request, collected: list):
    @tool
    def retrieve_user_to_item_candidates() -> str:
        """Find catalog tracks matching a listener's published CF-BPR factor directly.

        Only use when the listener context below gives a listener ID with a
        CF-BPR vector. Takes no arguments. Compares the listener's own factor
        directly against track factors, independent of other listeners.

        Returns:
            Catalog tracks by descending factor similarity, each with
            track_id, title, artists, tags; may be empty.
        """
        result = retriever.retrieve_user_to_item(request)
        collected.append(result)
        return as_tool_content(summarize_candidates(result))

    return retrieve_user_to_item_candidates
