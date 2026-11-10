"""Run with: python -m streamlit run app.py"""

import os

import streamlit as st

from music_recommender.config import Settings
from music_recommender.embeddings import LocalEmbedder
from music_recommender.graph import build_graph, create_llm
from music_recommender.retrieval import Retriever
from music_recommender.schemas import (
    ChatTurn,
    RecommendationRequest,
    RecommendationResponse,
    SimilarityWeights,
)
from music_recommender.store import CatalogStore

st.set_page_config(page_title="TalkPlay · Music discovery", page_icon="🎵", layout="wide")
settings = Settings.from_env()


@st.cache_resource
def get_store(path: str, model: str, column: str):
    from pathlib import Path

    return CatalogStore(
        Settings(data_dir=Path(path), embedding_model=model, track_embedding_column=column)
    )


@st.cache_resource
def get_embedder(model: str, device: str):
    local_settings = Settings.from_env()
    local_settings.embedding_model = model
    local_settings.embedding_device = device
    return LocalEmbedder(local_settings)


def render_response(response: RecommendationResponse):
    for position, rec in enumerate(response.recommendations, 1):
        with st.container(border=True):
            st.subheader(f"{position}. {rec.track.title}")
            st.text(", ".join(rec.track.artists) or "Artist unavailable")
            st.write(rec.explanation)
            with st.expander("Match details"):
                st.caption(f"Ranking score: {rec.score:.3f} · a relative score, not a probability")
                st.text("Track ID: " + rec.track.track_id)
                st.write("Catalog tags: " + ", ".join(rec.track.tags[:20]))
                if rec.supporting_users:
                    st.caption("These are predicted semantic affinities, not recorded likes.")
                    st.dataframe(
                        [
                            {
                                "Similar user": u.user_id,
                                "User similarity": round(u.similarity, 3),
                                "Track affinity": round(u.track_affinity, 3),
                            }
                            for u in rec.supporting_users
                        ],
                        hide_index=True,
                    )
    for notice in response.notices:
        st.warning(notice)
    if response.mode == "local":
        st.caption("Local retrieval mode · no Groq calls · template explanations")
    with st.expander("Inspect structured result"):
        st.json(response.model_dump())


st.title("Music for the moment")
st.write("Describe what you want to hear. Refine the suggestions as we go.")

with st.sidebar:
    st.header("Your listening session")
    catalog = st.selectbox("Catalog", ["Full catalog", "Sample catalog"])
    settings.data_dir /= "catalog" if catalog == "Full catalog" else "sample"
    local_only = st.checkbox(
        "Local retrieval only (no API calls)", value=not bool(os.getenv("GROQ_API_KEY"))
    )
    api_key = os.getenv("GROQ_API_KEY", "")
    if not local_only:
        api_key = st.text_input("Groq API key", value=api_key, type="password")
        settings.groq_model = st.text_input("Groq model", settings.groq_model)
        st.caption("Free-plan quotas apply to your Groq account. No automatic model switching.")

if not (settings.data_dir / "catalog.sqlite3").exists():
    st.info(
        "Build the catalog first. The sample is a quick setup check; use the full catalog to assess recommendation quality."
    )
    st.code(
        "python -m music_recommender.ingest --sample\n# Or the full catalog:\npython -m music_recommender.ingest",
        language="powershell",
    )
    st.stop()

try:
    store = get_store(
        str(settings.data_dir), settings.embedding_model, settings.track_embedding_column
    )
except Exception as exc:  # noqa: BLE001 - show catalog setup failures at the UI boundary
    st.error(f"Could not open this catalog: {exc}")
    st.stop()
if not store.state("ready"):
    st.info("Catalog ingestion is incomplete. Resume the ingestion command before chatting.")
    st.stop()
sessions = store.records("sessions")

with st.sidebar:
    st.caption(f"{store.collections['tracks'].count():,} searchable tracks")
    session_id = st.selectbox(
        "Starting profile",
        [None, *sessions.keys()],
        format_func=lambda key: (
            "New listener"
            if key is None
            else f"{sessions[key]['user_id'][:8]} · {sessions[key]['specificity']} · {key[:8]}"
        ),
    )
    selected = sessions.get(session_id, {})
    context_key = f"{catalog}:{session_id}"
    if st.session_state.get("context_key") != context_key:
        st.session_state.context_key = context_key
        st.session_state.chat = []
        st.session_state.seen = []
    if st.button("Start a new conversation"):
        st.session_state.chat = []
        st.session_state.seen = []
        st.rerun()
    profile = st.text_area(
        "Musical preferences / profile",
        value=selected.get("profile_text", ""),
        key=f"profile:{context_key}",
    )
    goal = st.text_area("Listening goal", value=selected.get("goal", ""), key=f"goal:{context_key}")
    specificities = [None, "HH", "HL", "LH", "LL"]
    specificity = st.selectbox(
        "Query specificity",
        specificities,
        index=specificities.index(selected.get("specificity")),
        format_func=lambda value: value or "Unspecified",
        key=f"spec:{context_key}",
        help="Dataset labels are compared by exact match. Unspecified omits this component.",
    )
    count = st.slider("Number of tracks", 1, 10, 5)
    avoid_repeats = st.checkbox("Avoid tracks already suggested", value=True)
    with st.expander("Similarity weights"):
        p = st.slider("User profile", 0.0, 1.0, 0.5, 0.05)
        g = st.slider("Conversation goal", 0.0, 1.0, 0.35, 0.05)
        s = st.slider("Specificity", 0.0, 1.0, 0.15, 0.05)
        query_weight = st.slider("Track score: request vs similar listeners", 0.0, 1.0, 0.75, 0.05)
        st.caption(
            "Weights are normalized over available components. Higher request weight gives similar listeners less influence."
        )

if catalog == "Sample catalog":
    st.caption(
        "Sample catalog: 300 source tracks and 24 training sessions. This checks the workflow, not recommendation quality."
    )

for entry in st.session_state.chat:
    with st.chat_message(entry["role"]):
        if "response" in entry:
            render_response(RecommendationResponse.model_validate(entry["response"]))
        else:
            st.write(entry["content"])

if prompt := st.chat_input("Try: mellow instrumental music for a rainy evening"):
    if p + g + s == 0:
        st.error("Choose at least one nonzero similarity weight.")
        st.stop()
    if not local_only and not api_key:
        st.error("Enter your Groq key or select local retrieval mode.")
        st.stop()
    request = RecommendationRequest(
        message=prompt,
        profile=profile,
        goal=goal,
        specificity=specificity,
        session_id=session_id,
        user_id=selected.get("user_id"),
        history=[
            ChatTurn(role=entry["role"], content=entry["content"])
            for entry in st.session_state.chat[-6:]
        ],
        excluded_track_ids=st.session_state.seen if avoid_repeats else [],
        weights=SimilarityWeights(profile=p, goal=g, specificity=s),
        query_weight=query_weight,
        limit=count,
    )
    with st.chat_message("user"):
        st.write(prompt)
    try:
        with st.spinner("Finding tracks and matching listener profiles…"):
            embedder = get_embedder(settings.embedding_model, settings.embedding_device)
            retriever = Retriever(settings, store, embedder)
            llm = None if local_only else create_llm(settings, api_key)
            result = build_graph(settings, retriever, llm).invoke(
                {"request": request, "notices": []}
            )
            response = result["response"]
    except Exception as exc:
        st.error(
            f"Recommendation failed ({type(exc).__name__}). Check the catalog and local model setup in the terminal."
        )
        import logging

        logging.getLogger(__name__).exception("Local recommendation pipeline failed")
        st.stop()
    summary = "\n".join(
        f"{r.track.title} by {', '.join(r.track.artists)}: {r.explanation}"
        for r in response.recommendations
    )
    st.session_state.chat.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": summary, "response": response.model_dump()},
        ]
    )
    st.session_state.seen = list(
        dict.fromkeys(
            [*st.session_state.seen, *(r.track.track_id for r in response.recommendations)]
        )
    )
    st.rerun()
