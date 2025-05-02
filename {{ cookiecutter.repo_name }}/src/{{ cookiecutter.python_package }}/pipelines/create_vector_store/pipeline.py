from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_embedding_function,
    create_vector_store,
    format_dialogs,
    select_vector_store,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=format_dialogs,
                inputs=["dialogs"],
                outputs="formatted_dialogs",
                name="format_dialogs_node",
            ),
            node(
                func=create_embedding_function,
                inputs=[],
                outputs="embedding_function",
                name="create_embedding_function_node",
                tags="agent_rag"
            ),
            node(
                func=select_vector_store,
                inputs=[
                    "params:vector_store_type",
                    "deeplake_vector_store_init",
                    "pinecone_vector_store_init",
                ],
                outputs="selected_vector_store_init",
                name="select_vector_store_init_node",
            ),
            node(
                func=create_vector_store,
                inputs=[
                    "selected_vector_store_init",
                    "formatted_dialogs",
                    "embedding_function",
                    "params:embedding_size",
                ],
                outputs="vector_store",
                name="create_vector_store_node",
            ),
        ]
    )
