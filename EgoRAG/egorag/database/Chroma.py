import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
import torch

try:
    import chromadb
    from chromadb.utils.data_loaders import ImageLoader
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
except ImportError:
    raise ImportError(
        "Chroma is not installed. Please install it with 'pip install chromadb'."
    )


class ChromaError(Exception):
    """Base exception for Chroma related errors."""
    pass


class DocumentNotFoundError(ChromaError):
    """Raised when a document is not found in the collection."""
    pass


class MetadataDict(TypedDict, total=False):
    date: str
    start_time: str
    end_time: str


class Chroma(ABC):
    def __init__(self, name: str = "EgoRAG", method: str = "text", db_dir: str = "./chroma_db"):
        self.name = name
        self.method = method
        self.db_dir = db_dir
        self.db_path = os.path.join(self.db_dir, self.name)
        # Initialize client
        self._client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = OpenCLIPEmbeddingFunction()

        # Get or create collection
        try:
            self._collection = self._client.get_collection(
                name=self.name,
                embedding_function=self.embedding_function,
                data_loader=ImageLoader(),
            )
            print(f"Find target collection {self.name}")
        except:
            self._collection = self._client.create_collection(
                name=self.name,
                embedding_function=self.embedding_function,
                data_loader=ImageLoader(),
            )
            print(f"Create new collection {self.name}")

    @property
    def client(self):
        return self._client

    @property
    def collection(self):
        return self._collection

    def _generate_id_range(self, id: str, n: int) -> List[str]:
        """Generate a range of IDs based on the given ID and range size."""
        try:
            parts = id.split("_")
            base_id = "_".join(parts[:-1])
            index = int(parts[-1])

            start_index = max(0, index - n)
            end_index = index + n

            return [f"{base_id}_{i}" for i in range(start_index, end_index + 1)]
        except (ValueError, IndexError) as e:
            raise ChromaError(f"Invalid ID format: {e}")

    def get_content_by_id(self, id: str) -> Optional[str]:
        try:
            result = self._collection.get(ids=[id])
            if result and result["documents"]:
                return result["documents"][0]
            raise DocumentNotFoundError(f"No content found for ID: {id}")
        except Exception as e:
            raise ChromaError(f"Error retrieving content by ID: {e}")

    def get_caption(self, id: str, n_result: int = 0) -> Optional[dict]:
        try:
            ids = self._generate_id_range(id=id, n=n_result)
            result = self._collection.get(ids=ids)
            
            if result and result["documents"]:
                return {
                    "item": {
                        "ids": [id],
                        "documents": result["documents"],
                        "metadatas": [result["metadatas"][0]],
                    },
                    "documents": result["documents"],
                }
            raise DocumentNotFoundError(f"No content found for ID: {id}")
        except ChromaError:
            raise
        except Exception as e:
            raise ChromaError(f"Error retrieving caption: {e}")

    def view_database(self, n: Optional[int] = None, show_embeddings: bool = False) -> None:
        try:
            all_data = self._collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            total_entries = len(all_data["ids"])
            print(f"Number of entries in the database: {total_entries}")
            
            documents_to_view = all_data["documents"][:n] if n is not None else all_data["documents"]
            
            for idx, document in enumerate(documents_to_view):
                metadata = all_data["metadatas"][idx] if "metadatas" in all_data else "No metadata"
                metadata_types = {key: type(value).__name__ for key, value in metadata.items()} if metadata != "No metadata" else "No metadata"
                
                output = f"ID: {all_data['ids'][idx]}, Content: {document}, Metadata: {metadata}, Metadata Types: {metadata_types}"
                if show_embeddings and "embeddings" in all_data:
                    output += f", Embedding shape: {len(all_data['embeddings'][idx])}"
                print(output)
        except Exception as e:
            raise ChromaError(f"Error viewing database: {e}")

    def get_doc(self, n: Optional[int] = None) -> List[Dict[str, Union[str, MetadataDict]]]:
        all_docs = []
        try:
            all_data = self._collection.get()
            print(f"Number of entries in the database: {len(all_data['ids'])}")
            
            documents_to_view = all_data["documents"][:n] if n is not None else all_data["documents"]
            
            for idx, document in enumerate(documents_to_view):
                metadata = all_data["metadatas"][idx] if "metadatas" in all_data else {}
                metadata_filtered: MetadataDict = {
                    key: metadata.get(key, "") 
                    for key in ["date", "end_time", "start_time"]
                    if key in metadata
                }
                all_docs.append({"Content": document, "Metadata": metadata_filtered})
                
            return all_docs
        except Exception as e:
            raise ChromaError(f"Error getting documents: {e}")

    def custom_query(
        self,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict] = None,
        filter_first: bool = True,
    ) -> dict:
        """
        Custom query method, supports direct retrieval from filtered data
        """
        try:
            if not filter_first or not where:
                return self._collection.query(
                    query_texts=query_texts, n_results=n_results, where=where
                )

            filtered_data = self._collection.get(
                where=where,
                include=["embeddings", "documents", "metadatas"],
            )

            if not filtered_data["ids"]:
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}

            query_embeddings = self.embedding_function(query_texts)
            filtered_embeddings = (
                np.array([self.embedding_function([doc])[0] for doc in filtered_data["documents"]])
                if filtered_data["embeddings"] is None
                else np.array(filtered_data["embeddings"])
            )

            similarities = np.dot(query_embeddings, filtered_embeddings.T) / (
                np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis]
                * np.linalg.norm(filtered_embeddings, axis=1)
            )

            n_results = min(n_results, len(filtered_data["ids"]))
            top_indices = np.argsort(similarities[0])[-n_results:][::-1]

            return {
                "ids": [filtered_data["ids"][i] for i in top_indices],
                "documents": [filtered_data["documents"][i] for i in top_indices],
                "metadatas": [filtered_data["metadatas"][i] for i in top_indices],
                "distances": [1 - similarities[0][i] for i in top_indices],
            }

        except Exception as e:
            raise ChromaError(f"Error in custom query: {e}")

    def clean_database(self):
        """
        Clear all data in the current database
        """
        try:
            self._collection.delete(self._collection.get()["ids"])
            print(f"All data in the collection {self.name} has been deleted.")
        except Exception as e:
            print(f"Error cleaning the database: {e}")

    def check_metadata_completeness(self, required_keys=None):
        """
        Check if each entry in the database contains the specified metadata keys
        Args:
            required_keys (list): List of metadata keys to check, if None, only check for the presence of metadata
        Returns:
            dict: Dictionary containing the check results
        """
        try:
            all_data = self._collection.get()
            total_entries = len(all_data["ids"])
            missing_metadata = []
            metadata_stats = {
                "entries_with_metadata": 0,
                "entries_without_metadata": 0,
                "metadata_key_frequency": {},
            }

            if total_entries == 0:
                return {"status": "empty", "message": "No data in the database"}

            # Count all metadata keys
            all_keys = set()
            for metadata in all_data["metadatas"]:
                if metadata != "No metadata" and metadata is not None:
                    all_keys.update(metadata.keys())

            # Initialize key frequency statistics
            for key in all_keys:
                metadata_stats["metadata_key_frequency"][key] = 0

            for idx, metadata in enumerate(all_data["metadatas"]):
                if metadata == "No metadata" or metadata is None:
                    metadata_stats["entries_without_metadata"] += 1
                    missing_metadata.append(
                        {"id": all_data["ids"][idx], "issue": "No metadata"}
                    )
                    continue

                metadata_stats["entries_with_metadata"] += 1
                # Count the frequency of each key
                for key in metadata:
                    metadata_stats["metadata_key_frequency"][key] += 1

                if required_keys:
                    missing_keys = [key for key in required_keys if key not in metadata]
                    if missing_keys:
                        missing_metadata.append(
                            {
                                "id": all_data["ids"][idx],
                                "issue": f"Missing keys: {missing_keys}",
                            }
                        )

            # Calculate the coverage percentage for each key
            for key in metadata_stats["metadata_key_frequency"]:
                frequency = metadata_stats["metadata_key_frequency"][key]
                coverage = (frequency / total_entries) * 100
                metadata_stats["metadata_key_frequency"][key] = {
                    "count": frequency,
                    "coverage_percentage": round(coverage, 2),
                }

            result = {
                "total_entries": total_entries,
                "entries_with_issues": len(missing_metadata),
                "all_complete": len(missing_metadata) == 0,
                "metadata_statistics": metadata_stats,
                "issues": missing_metadata if missing_metadata else None,
            }

            return result

        except Exception as e:
            print(f"Error checking metadata: {e}")
            return {"status": "error", "message": str(e)}

    def check_and_clean_empty_entries(self):
        """
        Check all entries and delete entries with empty documents or metadatas.
        Returns a dictionary containing the list of deleted IDs, count, and status information.
        """
        try:
            # Get all data, including documents and metadatas
            all_data = self._collection.get(include=["documents", "metadatas"])
            ids = all_data["ids"]
            documents = all_data["documents"]
            metadatas = all_data["metadatas"]

            ids_to_delete = []
            for idx in range(len(ids)):
                doc = documents[idx]
                meta = metadatas[idx] if idx < len(metadatas) else None

                # Check if document is empty (None or empty string)
                doc_empty = doc is None or (isinstance(doc, str) and doc.strip() == "")
                # Check if metadata is empty (None or empty dictionary)
                meta_empty = meta is None or meta == {}

                if doc_empty or meta_empty:
                    ids_to_delete.append(ids[idx])

            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                print(
                    f"Deleted {len(ids_to_delete)} entries with empty documents or metadatas."
                )
                return {
                    "deleted_ids": ids_to_delete,
                    "count": len(ids_to_delete),
                    "message": f"Successfully deleted {len(ids_to_delete)} entries.",
                }
            else:
                print("No empty entries found.")
                return {
                    "deleted_ids": [],
                    "count": 0,
                    "message": "All entries are valid. No deletions needed.",
                }
        except Exception as e:
            print(f"Error checking and cleaning empty entries: {e}")
            return {
                "error": str(e),
                "message": "An error occurred during the cleaning process.",
            }


if __name__ == "__main__":
    a = Chroma(name="EgoRAG")
