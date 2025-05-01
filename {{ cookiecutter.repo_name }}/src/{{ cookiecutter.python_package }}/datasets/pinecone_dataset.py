"""Pinecone vector database dataset for Kedro."""
from typing import Any, NoReturn, Optional, List, Dict

from kedro.io import AbstractDataset, DatasetError


class PineconeVectorStoreDataset(AbstractDataset):
    """A Kedro dataset for interacting with Pinecone Vector Database.

    This dataset provides an interface for loading and interacting with a Pinecone index,
    which can be used for vector similarity search in production environments.
    """

    def __init__(
            self,
            api_key: str,
            environment: str,
            index_name: str,
            namespace: Optional[str] = None,
            dimension: int = 1536,
            metric: str = "cosine",
            **kwargs,
    ):
        """Initializes the dataset with Pinecone credentials and index information.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Optional namespace within the index for scoping operations
            dimension: Dimension of the vectors (default is 1536 for OpenAI embeddings)
            metric: Distance metric to use (default is cosine)
            **kwargs: Additional arguments for the Pinecone client
        """
        self._api_key = api_key
        self._environment = environment
        self._index_name = index_name
        self._namespace = namespace
        self._dimension = dimension
        self._metric = metric
        self._kwargs = kwargs or {}
        self._client = None
        self._index = None

    def _connect(self):
        """Connect to Pinecone and get or create the index."""
        try:
            # Import here to avoid requiring pinecone-client for users who don't use it
            from pinecone import Pinecone, PodSpec

            if self._client is None:
                self._client = Pinecone(api_key=self._api_key)

            # Check if index exists, create if it doesn't
            index_list = [idx.name for idx in self._client.list_indexes()]

            if self._index_name not in index_list:
                self._client.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric=self._metric,
                    spec=PodSpec(environment=self._environment)
                )

            self._index = self._client.Index(self._index_name)
            return self._index

        except ImportError:
            raise ImportError(
                "Could not import pinecone-client. "
                "Make sure it is installed by running: pip install pinecone-client>=2.0.0"
            )
        except Exception as e:
            raise DatasetError(f"Error connecting to Pinecone: {str(e)}")

    def load(self) -> Any:
        """Loads and returns the Pinecone Index.

        Returns:
            A wrapped Pinecone index with additional helper methods to 
            maintain a consistent interface with other vector stores.
        """
        index = self._connect()

        # Create a wrapper with a compatible interface to DeepLake
        class PineconeWrapper:
            def __init__(self, index, namespace=None):
                self.index = index
                self.namespace = namespace

            def add(self, text: List[str], embedding: List[List[float]], metadata: List[Dict] = None):
                """Add items to the vector store with a DeepLake-compatible interface."""
                if metadata is None:
                    metadata = [{}] * len(text)

                vectors = []
                for i, (txt, emb, meta) in enumerate(zip(text, embedding, metadata)):
                    # Ensure embedding is a list
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()

                    # Create a vector entry
                    meta["text"] = txt
                    vector = {
                        "id": f"vec_{i}_{hash(txt) % 10000}",
                        "values": emb,
                        "metadata": meta
                    }
                    vectors.append(vector)

                # Upsert in batches of 100
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    self.index.upsert(vectors=batch, namespace=self.namespace)

                return len(vectors)

            def search(self, embedding_data: str, embedding_function: callable, k: int = 1):
                """Search the vector store with a DeepLake-compatible interface."""
                # Convert query to embedding
                query_embedding = embedding_function([embedding_data])[0]
                if hasattr(query_embedding, "tolist"):
                    query_embedding = query_embedding.tolist()

                # Perform the query
                results = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True,
                    namespace=self.namespace
                )

                texts = []
                if results.matches:
                    texts = [match.metadata.get("text", "") for match in results.matches]

                return {"text": texts, "metadata": results.matches}

        return PineconeWrapper(index, self._namespace)

    def save(self, data: None) -> NoReturn:
        """Pinecone indexes are managed through their API.

        This method doesn't support direct saving as the index is created during initialization.

        Args:
            data: This argument is unused.

        Raises:
            DatasetError: Always raised since direct saving is not supported.
        """
        raise DatasetError(f"{self.__class__.__name__} does not support direct saving. "
                           "The index is created during initialization.")

    def _describe(self) -> dict[str, Any]:
        """Returns a dictionary describing the dataset configuration.

        Returns:
            A dictionary containing the dataset configuration.
        """
        return {
            "index_name": self._index_name,
            "environment": self._environment,
            "namespace": self._namespace,
            "dimension": self._dimension,
            "metric": self._metric,
            **self._kwargs
        }