"""Small Streamlit UI for the injectable financial-document pipeline."""

import uuid

import streamlit as st
import logging

from src.config import AppConfig
from src.data import load_saved_sample
from src.pipeline import build_strategies_from_config, run_pipeline
from src.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


@st.cache_resource
def load_app():
    logger.info("APPLICATION_INITIALIZATION_STARTED")
    config = AppConfig.from_env()
    logger.info(
        "CONFIG_LOADED | search=%s | rag=%s | answer=%s | prompt=%s",
        config.search_strategy,
        config.rag_strategy,
        config.answer_strategy,
        config.prompt_strategy,
    )
    documents = load_saved_sample()

    model_callable = None

    if config.answer_strategy == "llm":
        from src.generate.generate_answer import build_model_callable

        logger.info("LLM_INITIALIZATION_STARTED")
        model_callable = build_model_callable()
        logger.info("LLM_INITIALIZATION_COMPLETED")

    strategies = build_strategies_from_config(
        config,
        model_callable=model_callable,
    )

    logger.info(
        "PIPELINE_INITIALIZED | documents=%d | answerer=%s | rag=%s",
        len(documents),
        type(strategies.answerer).__name__,
        type(strategies.rag).__name__,
    )

    logger.info("APPLICATION_INITIALIZATION_COMPLETED")

    return documents, strategies, config


def main():
    st.set_page_config(page_title="Financial Document Copilot", page_icon="📄")
    st.title("Financial Document Copilot")
    st.caption("Answers are retrieved from a 10-document sample; this is a project demo.")

    try:
        documents, strategies, config = load_app()
    except FileNotFoundError:
        st.error("Sample data is missing. Run `uv run python -m scripts.download_sample`.")
        return
    except ValueError as exc:
        st.error(f"Invalid strategy config: {exc}")
        return

    st.write(f"Loaded {len(documents)} sample documents.")
    st.caption(
        f"Config: search={config.search_strategy}, rag={config.rag_strategy}, "
        f"prompt={config.prompt_strategy}, answer={config.answer_strategy}"
    )
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    question = st.text_input("Ask a question about the documents")
    uploaded_image = st.file_uploader("Optional image", type=["png", "jpg", "jpeg"])
    if st.button("Ask") and question:
        image = uploaded_image.getvalue() if uploaded_image else None
        logger.info(
            "USER_QUERY_SUBMITTED | question_length=%d | session_id=%s",
            len(question),
            st.session_state.session_id,
        )
        try:
            result = run_pipeline(
                question,
                documents,
                session_id=st.session_state.session_id,
                image=image,
                strategies=strategies,
            )
            logger.info("UI_QUERY_COMPLETED")
        except ValueError as exc:
            st.warning(str(exc))
            return
        except Exception:
            logger.exception("UI_QUERY_FAILED")
            st.error("The query failed. Check logs/application.log for details.")
            return

        st.subheader("Answer")
        st.write(result["answer"]["answer"])
        if result["answer"]["should_refuse"]:
            st.info(result["answer"]["reason"])
        st.caption(f"Citations: {result['answer']['citations'] or 'None'}")
        with st.expander("Pipeline details"):
            st.write("Conversation history", result["history"])
            st.write(" → ".join(result["steps"]))
            st.json(result["metrics"])
            st.write("Retrieved evidence")

            for index, item in enumerate(result["evidence"], start=1):
                st.markdown(
                    f"**{index}. {item.doc_id}** | "
                    f"Chunk ID: `{item.chunk_id or 'Not available'}`"
                )

                st.caption(
                    f"Relevance score: {item.score:.4f}"
                )

                st.write(f"**Question:** {item.question}")
                st.write(f"**Answer:** {item.answer}")
                st.write("Chunk character count:", len(item.text))
                with st.expander("View retrieved chunk text"):
                    st.code(item.text, language="text")


if __name__ == "__main__":
    main()
