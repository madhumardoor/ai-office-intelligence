"""Import every model so Base.metadata is complete (Alembic autogenerate + tests rely on this)."""

from app.models.app_data import ApiKey, Conversation, Message, User
from app.models.company import (
    Company,
    CompanyAlias,
    CompanyEmployeeSnapshot,
    CompanyFunding,
    CompanyHiringSnapshot,
    CompanyIdentifier,
    CompanyLocation,
    CompanyNews,
    EntityMatchCandidate,
)
from app.models.coworking import (
    CoworkingBranch,
    CoworkingBuilding,
    CoworkingOperator,
    CoworkingTenant,
    Property,
)
from app.models.knowledge import Document, DocumentChunk, EmbeddingMetadata
from app.models.ops import GeocodeCache, LLMLog, SearchLog, WebSearchCache
from app.models.provenance import IngestionRun, Source
from app.models.signals import CompanySignalScore, Signal, SignalDefinition

__all__ = [
    "ApiKey", "Company", "CompanyAlias", "CompanyEmployeeSnapshot", "CompanyFunding",
    "CompanyHiringSnapshot", "CompanyIdentifier", "CompanyLocation", "CompanyNews",
    "CompanySignalScore", "Conversation", "CoworkingBranch", "CoworkingBuilding",
    "CoworkingOperator", "CoworkingTenant", "Document", "DocumentChunk", "EmbeddingMetadata",
    "EntityMatchCandidate", "GeocodeCache", "IngestionRun", "LLMLog", "Message", "Property",
    "SearchLog", "Signal", "SignalDefinition", "Source", "User", "WebSearchCache",
]