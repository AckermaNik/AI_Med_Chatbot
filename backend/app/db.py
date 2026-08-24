"""Async engine and session factory."""

# AsyncIterator is just a type hint. It tells Python that our get_session function 
# is going to yield data asynchronously over time.
from collections.abc import AsyncIterator

# These are the heavy lifters from SQLAlchemy (our database library).
# We use the 'asyncio' versions because modern apps handle many users at once 
# without waiting in line (asynchronous programming).
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

# We run the function to load our .env and Python defaults into memory.
settings = get_settings()

# --- THE ENGINE (Driver) ---
# We create the core connection to PostgreSQL.
engine = create_async_engine(
    # This uses the string we saw earlier: "postgresql+asyncpg://medai:medai@..."
    settings.database_url,
    
    # echo=False means "Don't print every single SQL query to the terminal."
    # If you ever have a bug and want to see the raw SQL, change this to True!
    echo=False,
    
    # This is a lifesaver. Before the engine sends a query, it "pings" the database 
    # to make sure the connection hasn't timed out or dropped. If it has, it reconnects automatically.
    pool_pre_ping=True,
)

# --- THE SESSION FACTORY ---
# We create a factory that will generate our database sessions.
SessionLocal = async_sessionmaker(
    # We bind the sessions to the engine we just created above.
    engine,
    
    # We specify that we want Async waiters, not standard synchronous ones.
    class_=AsyncSession,
    
    # When you save data (commit), SQLAlchemy normally "expires" the local Python object. 
    # If you try to read that object again, it crashes. Setting this to False keeps 
    # your Python data safe to read even after you've saved it to the database.
    expire_on_commit=False,
)

# --- THE FASTAPI DEPENDENCY ---
# This function is what your actual API routes will call when they need to talk to the DB.
async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    
    # The 'async with' block is brilliant. It tells Python:
    # 1. Ask the factory (SessionLocal) for a new session.
    # 2. Hand it to the user.
    # 3. IMPORTANT: When the user is done, AUTOMATICALLY close the connection! 
    # This prevents memory leaks and database crashes.
    async with SessionLocal() as session:
        # 'yield' means "pause here, hand the session to the API route, and wait 
        # until the API route is totally finished before moving on and closing it."
        yield session