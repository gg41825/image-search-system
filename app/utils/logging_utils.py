import asyncio
import traceback
import functools
from starlette.datastructures import UploadFile
from db.mongo_client import MongoDBHandler


def sanitize_for_mongo(data):
    """Recursively convert non-serializable objects (like UploadFile) into safe dicts."""
    if isinstance(data, UploadFile):
        return {
            "filename": data.filename,
            "content_type": data.content_type,
            "size": getattr(data.file, "len", None)
        }
    elif isinstance(data, (list, tuple, set)):
        return [sanitize_for_mongo(v) for v in data]
    elif isinstance(data, dict):
        return {str(k): sanitize_for_mongo(v) for k, v in data.items()}
    elif isinstance(data, bytes):
        return f"<bytes:{len(data)}>"
    elif hasattr(data, "__dict__"):
        return sanitize_for_mongo(vars(data))
    else:
        try:
            str(data)
            return data
        except Exception:
            return repr(data)


def with_logging(event_name: str):
    """Decorator for automatic MongoDB logging of function calls."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            db = MongoDBHandler()
            try:
                result = await func(*args, **kwargs)
                db.log_event(event_name, {
                    "status": "success",
                    "args": sanitize_for_mongo(args),
                    "kwargs": sanitize_for_mongo(kwargs),
                })
                return result
            except Exception as e:
                db.log_error(type(e).__name__, str(e), traceback.format_exc())
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            db = MongoDBHandler()
            try:
                result = func(*args, **kwargs)
                db.log_event(event_name, {
                    "status": "success",
                    "args": sanitize_for_mongo(args),
                    "kwargs": sanitize_for_mongo(kwargs),
                })
                return result
            except Exception as e:
                db.log_error(type(e).__name__, str(e), traceback.format_exc())
                raise

        # Choose correct wrapper based on whether the target is async
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator