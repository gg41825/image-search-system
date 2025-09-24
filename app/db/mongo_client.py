import datetime
from typing import List, Dict, Optional
from pymongo import MongoClient, errors

import config

class MongoDBHandler:
    """MongoDB handler for managing products and logs collections."""

    def __init__(self, uri: str = config.MONGO_URI, db_name: str = config.MONGO_DB_NAME):
        """
        Initialize MongoDB connection.

        Args:
            uri (str): MongoDB connection URI.
            db_name (str): Database name.
        """
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")  # Verify connection
        except errors.ServerSelectionTimeoutError as e:
            raise RuntimeError(f"❌ Could not connect to MongoDB: {e}")

        self.db = self.client[db_name]
        self.products = self.db["products"]
        self.logs = self.db["logs"]

    # ---------- Product Methods ----------
    def insert_products(self, metadata_list: List[Dict]) -> List:
        """
        Insert multiple product documents.

        Args:
            metadata_list (List[Dict]): List of product documents.

        Returns:
            List: Inserted product IDs.
        """
        if not metadata_list:
            raise ValueError("metadata_list is empty, nothing to insert.")
        result = self.products.insert_many(metadata_list)
        return result.inserted_ids

    def get_product(self, product_id: int) -> Optional[Dict]:
        """
        Find product by its id.

        Args:
            product_id (int): Product ID.

        Returns:
            Optional[Dict]: Product document if found, else None.
        """
        return self.products.find_one({"id": product_id})
    
    def get_sample_ids(self, sample_size: int = 50) -> List[int]:
        """
        Randomly sample product IDs from the collection.

        Args:
            sample_size (int): Number of product IDs to sample.

        Returns:
            List[int]: List of randomly sampled product IDs.
        """
        if sample_size <= 0:
            # Return all product IDs
            sample_docs = list(self.products.find({}, {"id": 1}))
        else:
            sample_docs = list(self.products.aggregate([
                {"$sample": {"size": sample_size}},
                {"$project": {"id": 1}}
            ]))
        return [doc["id"] for doc in sample_docs]

    # ---------- Logging Methods ----------
    def _log(self, event: str, details: Dict) -> str:
        """
        Internal helper to insert a log document.

        Args:
            event (str): Event type (e.g., "error", "info").
            details (Dict): Log details.

        Returns:
            str: Inserted log document ID.
        """
        doc = {
            "timestamp": datetime.datetime.now(),
            "event": event,
            **details,
        }
        result = self.logs.insert_one(doc)
        return str(result.inserted_id)

    def log_error(self, error_type: str, message: str, stacktrace: str) -> str:
        """
        Log an error event.

        Args:
            error_type (str): Type of error.
            message (str): Error message.
            stacktrace (str): Stacktrace details.

        Returns:
            str: Inserted log document ID.
        """
        return self._log("error", {
            "type": error_type,
            "message": message,
            "stacktrace": stacktrace,
        })

    def log_event(self, event_type: str, details: Dict) -> str:
        """
        Log a general event.

        Args:
            event_type (str): Type of event.
            details (Dict): Event details.

        Returns:
            str: Inserted log document ID.
        """
        return self._log(event_type, {"details": details})
