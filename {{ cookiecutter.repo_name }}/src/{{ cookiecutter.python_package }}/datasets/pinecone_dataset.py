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
            credentials: Optional[dict] = None,
            dimension: int = 16384,
            metric: str = "cosine",
            namespace: Optional[str] = None,
            api_key: Optional[str] = None,
            environment: Optional[str] = None,
            index_name: Optional[str] = None,
            **kwargs,
    ):
        """Initializes the dataset with Pinecone credentials and index information.

        Args:
            credentials: Dictionary containing api_key, environment, and index_name
            dimension: Dimension of the vectors
            metric: Distance metric to use (default is cosine)
            namespace: Optional namespace within the index for scoping operations
            api_key: Optional direct API key (overrides credentials)
            environment: Optional direct environment (overrides credentials)
            index_name: Optional direct index name (overrides credentials)
            **kwargs: Additional arguments for the Pinecone client
        """
        # Get credentials from either the credentials dict or direct params
        credentials = credentials or {}
        self._api_key = api_key or credentials.get("api_key")
        self._environment = environment or credentials.get("environment")
        self._index_name = index_name or credentials.get("index_name")

        # Validate required credentials
        if not self._api_key:
            raise ValueError("Pinecone API key is required")
        if not self._environment:
            raise ValueError("Pinecone environment is required")
        if not self._index_name:
            raise ValueError("Pinecone index name is required")

        self._namespace = namespace or "default"
        self._dimension = dimension
        self._metric = metric
        self._kwargs = kwargs or {}
        self._client = None
        self._index = None

    def _connect(self):
        """Connect to Pinecone and get or create the index.

        Uses the Pinecone v2 API pattern.
        """
        try:
            # Import here to avoid requiring pinecone-client for users who don't use it
            from pinecone import Pinecone

            # Create Pinecone client using v2 API
            self._client = Pinecone(api_key=self._api_key)

            # Check if index exists
            existing_indexes = self._client.list_indexes()
            index_exists = any(idx.name == self._index_name for idx in existing_indexes)

            if not index_exists:
                print(f"Creating new index: {self._index_name}")
                # Create a new index
                self._client.create_index(
                    name=self._index_name,
                    dimension=self._dimension,
                    metric=self._metric,
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": self._environment
                        }
                    }
                )

            # Get the index
            self._index = self._client.Index(self._index_name)
            return self._index

        except ImportError:
            raise ImportError(
                "Could not import pinecone-client v2.x. "
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
            def __init__(self, index, namespace=None, dimension=16384):
                self.index = index
                self.namespace = namespace
                self.dimension = dimension

            def add(self, text: List[str], embedding: List[List[float]], metadata: List[Dict] = None):
                """Add items to the vector store with a DeepLake-compatible interface."""
                if metadata is None:
                    metadata = [{}] * len(text)

                records = []
                for i, (txt, emb, meta) in enumerate(zip(text, embedding, metadata)):
                    # Ensure embedding is a list
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()

                    # Add text to metadata
                    meta = dict(meta)  # Make a copy to avoid modifying the original
                    meta["text"] = txt

                    # Create a record with Pinecone v2 format - using 'id' instead of '_id'
                    records.append({
                        "id": f"vec_{i}_{hash(txt) % 10000}",  # Changed from '_id' to 'id'
                        "values": emb,
                        "metadata": meta
                    })

                # Upsert in batches of 100
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    try:
                        # First try the direct upsert method
                        self.index.upsert(
                            vectors=batch,
                            namespace=self.namespace
                        )
                    except Exception as e:
                        print(f"Warning: First upsert attempt failed with error: {str(e)}. Trying alternate format...")
                        try:
                            # Try the records format as a fallback
                            self.index.upsert_records(
                                namespace=self.namespace,
                                records=batch
                            )
                        except Exception as e2:
                            print(f"Warning: Second upsert attempt also failed: {str(e2)}. Check Pinecone configuration.")
                            raise

                return len(records)

            def search(self, embedding_data: str, embedding_function: callable, k: int = 1):
                """Search the vector store with a DeepLake-compatible interface."""
                # Convert query to embedding - ensure correct dimensionality
                query_embedding = embedding_function([embedding_data], self.dimension)[0]
                if hasattr(query_embedding, "tolist"):
                    query_embedding = query_embedding.tolist()

                try:
                    # Try modern query API first
                    results = self.index.query(
                        namespace=self.namespace,
                        top_k=k,
                        include_metadata=True,
                        vector=query_embedding,
                    )

                    # Process results
                    texts = []
                    metadata = []

                    # Handle different result formats
                    if hasattr(results, 'matches') and results.matches:
                        texts = [match.metadata.get("text", "") for match in results.matches if hasattr(match, 'metadata')]
                        metadata = results.matches
                    elif isinstance(results, dict) and 'matches' in results:
                        texts = [match.get('metadata', {}).get("text", "") for match in results['matches']]
                        metadata = results['matches']

                    if not texts:
                        texts = ["No relevant context found in Pinecone index"]

                    return {"text": texts, "metadata": metadata}

                except Exception as e:
                    print(f"Warning: Initial search attempt failed: {str(e)}. Trying alternate format...")

                    try:
                        # Try with query_vector param instead
                        results = self.index.query(
                            namespace=self.namespace,
                            top_k=k,
                            include_metadata=True,
                            query_vector=query_embedding
                        )

                        # Extract text from results
                        texts = []
                        metadata = []

                        if hasattr(results, 'matches') and results.matches:
                            texts = [match.metadata.get("text", "") for match in results.matches if hasattr(match, 'metadata')]
                            metadata = results.matches
                        elif isinstance(results, dict) and 'matches' in results:
                            texts = [match.get('metadata', {}).get("text", "") for match in results['matches']]
                            metadata = results['matches']

                        if not texts:
                            texts = ["No relevant context found in Pinecone index"]

                        return {"text": texts, "metadata": metadata}

                    except Exception as e2:
                        print(f"Warning: All search attempts failed: {str(e2)}. Returning empty results.")
                        return {"text": ["Error querying Pinecone index"], "metadata": []}

        return PineconeWrapper(index, self._namespace, self._dimension)

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
