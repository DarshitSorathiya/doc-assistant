# ============================================================
# Database Engine & Session Management
# ============================================================
# Provides the SQLAlchemy engine, sessionmaker, and a FastAPI
# dependency (`get_db`) that yields a session per request.
# ============================================================

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from typing import Generator

from app.config.settings import get_settings


# ------------------------------------------------------------------
# Declarative base class — all models inherit from this
# ------------------------------------------------------------------
class Base(DeclarativeBase):
    """SQLAlchemy declarative base.

    Every ORM model (User, Document, Entity, etc.) inherits from this.
    `Base.metadata` holds the table definitions used by
    `create_all()` and Alembic migrations.
    """
    pass


# ------------------------------------------------------------------
# Engine & Session factory
# ------------------------------------------------------------------
settings = get_settings()

# SQLite needs special args for multi-threaded FastAPI access
_connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    echo=settings.DEBUG,           # log SQL in debug mode
    pool_pre_ping=True,            # detect stale connections
)

# Enable WAL mode and foreign keys for SQLite
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ------------------------------------------------------------------
# FastAPI dependency — yields a DB session per request
# ------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, guaranteed to close after the request.

    Usage in a route:

        @router.post("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------
# Utility: create all tables (dev convenience — use Alembic in prod)
# ------------------------------------------------------------------
def create_tables() -> None:
    """Create all tables defined by models that inherit from Base.

    Only use for local dev / tests. Production uses Alembic migrations.
    """
    Base.metadata.create_all(bind=engine)

    # ``create_all`` creates missing tables but does not add columns to existing
    # ones. This additive migration is needed by databases created before chat
    # document-selection metadata was introduced. It preserves existing chats.
    inspector = inspect(engine)
    if "chat_messages" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("chat_messages")}
        if "document_options" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE chat_messages ADD COLUMN document_options JSON")
                )
