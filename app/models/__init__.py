# ============================================================
# SQLAlchemy ORM Models
# ============================================================
# All tables for the Personal Document Finder.
# Models are kept in one file for simplicity during early phases;
# they can be split later if the file grows too large.
# ============================================================

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Enum,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship

from app.database import Base


# ------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------
class UserRole(str, enum.Enum):
    """Roles for RBAC: Admin > User > Viewer."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class DocumentStatus(str, enum.Enum):
    """Processing lifecycle of an uploaded document."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentCategory(str, enum.Enum):
    """Classification categories for documents."""
    INSURANCE = "Insurance"
    BANKING = "Banking"
    GOVERNMENT_ID = "Government ID"
    PROPERTY = "Property"
    TAX = "Tax"
    MEDICAL = "Medical"
    EDUCATION = "Education"
    EMPLOYMENT = "Employment"
    LEGAL = "Legal"
    FINANCIAL = "Financial"
    UTILITY = "Utility"
    OTHER = "Other"


# ------------------------------------------------------------------
# Helper: UTC-aware timestamp
# ------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ==================================================================
# USER
# ==================================================================
class User(Base):
    """Registered user with hashed password and role."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(
        Enum(UserRole),
        default=UserRole.USER,
        nullable=False,
    )
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


# ==================================================================
# DOCUMENT
# ==================================================================
class Document(Base):
    """An uploaded document (PDF/image) with OCR text and metadata."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(500), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes

    # Classification
    category = Column(Enum(DocumentCategory), nullable=True)
    classification_confidence = Column(Float, nullable=True)
    classification_method = Column(String(50), nullable=True)  # "rule" | "llm" | "hybrid"

    # Processing state
    status = Column(
        Enum(DocumentStatus),
        default=DocumentStatus.UPLOADED,
        nullable=False,
    )
    page_count = Column(Integer, nullable=True)
    extracted_text = Column(Text, nullable=True)  # full OCR text (all pages)

    # Version tracking
    version = Column(Integer, default=1, nullable=False)
    current_version = Column(Boolean, default=True, nullable=False)

    # Ownership
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    owner = relationship("User", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    tags = relationship("Tag", back_populates="document", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "status"),
        Index("ix_documents_user_category", "user_id", "category"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"


# ==================================================================
# DOCUMENT VERSION
# ==================================================================
class DocumentVersion(Base):
    """Historical version of a document (created on re-upload)."""

    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number = Column(Integer, nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_hash = Column(String(64), nullable=False)
    file_size = Column(Integer, nullable=False)
    changes_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="versions")

    def __repr__(self) -> str:
        return f"<DocumentVersion doc={self.document_id} v={self.version_number}>"


# ==================================================================
# ENTITY
# ==================================================================
class Entity(Base):
    """An extracted entity (PAN, Aadhaar, date, amount, etc.)."""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String(100), nullable=False, index=True)  # "PAN", "Aadhaar", etc.
    value = Column(String(500), nullable=False)
    confidence = Column(Float, nullable=True)
    source_page = Column(Integer, nullable=True)

    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="entities")

    __table_args__ = (
        Index("ix_entities_type_value", "entity_type", "value"),
    )

    def __repr__(self) -> str:
        return f"<Entity type={self.entity_type} value={self.value[:20]}>"


# ==================================================================
# TAG
# ==================================================================
class Tag(Base):
    """User-defined or system-generated tag on a document."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="tags")

    def __repr__(self) -> str:
        return f"<Tag name={self.name}>"


# ==================================================================
# AUDIT LOG
# ==================================================================
class AuditLog(Base):
    """Immutable audit trail of every API/tool action."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action = Column(String(100), nullable=False)       # e.g., "login", "upload", "query"
    resource_type = Column(String(100), nullable=True)  # e.g., "document", "search"
    resource_id = Column(String(100), nullable=True)
    details = Column(JSON, nullable=True)               # arbitrary JSON payload
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_user_action", "user_id", "action"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog action={self.action} user={self.user_id}>"


# ==================================================================
# CHAT HISTORY
# ==================================================================
class ChatSession(Base):
    """Persisted conversation metadata owned by a single user."""

    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), nullable=False, default="New chat")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    owner = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessageRecord", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessageRecord.id")


class ChatMessageRecord(Base):
    """Persisted chat turn, including citations for rendering history."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    document_options = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")
