"""Versioned Phase 2 audit rows. Unknown observations are represented by null."""
from dataclasses import dataclass
from typing import ClassVar
from ..contracts import Metadata, ContractError, _identity, _relative, _hash, _text, _version


def _optional_text(value, name):
    _text(value, name, optional=True)


def _scope(value):
    if value not in ('fixture', 'research'):
        raise ContractError('scope must be fixture or research')


def _scoped_id(value, scope, name):
    _identity(value, name)
    if (value.split(':', 1)[0] == 'fixture') != (scope == 'fixture'):
        raise ContractError(f'{name} does not match {scope} scope')


def _count(value, name):
    if value is not None and (type(value) is not int or value < 0):
        raise ContractError(f'{name} must be a nonnegative integer or null')


@dataclass(frozen=True)
class FileInventory(Metadata):
    schema_name: ClassVar[str] = 'file_inventory'
    schema_version: int
    scope: str
    file_id: str
    source: str
    component: str
    relative_path: str
    file_type: str
    size_bytes: int | None
    scan_started_utc: str
    scan_finished_utc: str | None
    mtime_ns: int | None
    sha256: str | None
    failure: str | None
    state: str

    def __post_init__(self):
        _version(self.schema_version); _scope(self.scope); _scoped_id(self.file_id, self.scope, 'file_id')
        for name in ('source','component','file_type','scan_started_utc','state'): _text(getattr(self,name),name)
        _relative(self.relative_path); _count(self.size_bytes,'size_bytes'); _count(self.mtime_ns,'mtime_ns')
        _hash(self.sha256,'sha256', optional=True)
        for name in ('scan_finished_utc','failure'): _optional_text(getattr(self,name),name)
        if self.state not in ('pending','completed','error','unstable'): raise ContractError('invalid inventory state')
        if self.state == 'completed' and (self.sha256 is None or self.failure is not None): raise ContractError('completed file requires hash and no failure')
        if self.state in ('error','unstable') and not self.failure: raise ContractError('failure state requires evidence')


@dataclass(frozen=True)
class Provenance(Metadata):
    schema_name: ClassVar[str] = 'provenance'
    schema_version: int
    scope: str
    source: str
    local_snapshot: str
    publisher_revision: str | None
    publisher_match_state: str
    archive_origin: str | None
    archive_checksum: str | None
    document_refs: list[str]
    license_refs: list[str]
    evidence: str | None

    def __post_init__(self):
        _version(self.schema_version); _scope(self.scope)
        for name in ('source','local_snapshot','publisher_match_state'): _text(getattr(self,name),name)
        for name in ('publisher_revision','archive_origin','archive_checksum','evidence'): _optional_text(getattr(self,name),name)
        for name in ('document_refs','license_refs'):
            values = getattr(self,name)
            if not isinstance(values,list): raise ContractError(f'{name} must be a list')
            for value in values: _text(value,name)


@dataclass(frozen=True)
class MappingEvidence(Metadata):
    schema_name: ClassVar[str] = 'mapping_evidence'
    schema_version: int
    scope: str
    file_id: str
    field: str
    raw_value: str | None
    normalized_value: str | None
    verified: bool
    evidence: str | None

    def __post_init__(self):
        _version(self.schema_version); _scope(self.scope); _scoped_id(self.file_id,self.scope,'file_id'); _text(self.field,'field')
        for name in ('raw_value','normalized_value','evidence'): _optional_text(getattr(self,name),name)
        if type(self.verified) is not bool: raise ContractError('verified must be boolean')
        if self.verified and (self.normalized_value is None or self.evidence is None): raise ContractError('verified mapping requires value and evidence')


@dataclass(frozen=True)
class PairRelationship(Metadata):
    schema_name: ClassVar[str] = 'pair_relationship'
    schema_version: int
    scope: str
    endpoint_a: str
    endpoint_b: str
    relation_type: str
    evidence: str | None
    state: str
    reviewer: str | None
    disposition: str | None

    def __post_init__(self):
        _version(self.schema_version); _scope(self.scope)
        for name in ('endpoint_a','endpoint_b'): _scoped_id(getattr(self,name),self.scope,name)
        if self.endpoint_a == self.endpoint_b: raise ContractError('relationship endpoints must differ')
        _text(self.relation_type,'relation_type')
        for name in ('evidence','reviewer','disposition'): _optional_text(getattr(self,name),name)
        if self.state not in ('candidate','verified','rejected'): raise ContractError('invalid relationship state')
        if self.state != 'candidate' and not self.evidence: raise ContractError('reviewed relationship requires evidence')


@dataclass(frozen=True)
class Disposition(Metadata):
    schema_name: ClassVar[str] = 'disposition'
    schema_version: int
    scope: str
    subject_id: str
    reason_code: str
    stage: str
    state: str
    evidence: str | None
    date: str

    def __post_init__(self):
        _version(self.schema_version); _scope(self.scope); _scoped_id(self.subject_id,self.scope,'subject_id')
        for name in ('reason_code','stage','date'): _text(getattr(self,name),name)
        _optional_text(self.evidence,'evidence')
        if self.state not in ('retained','excluded','pending'): raise ContractError('invalid disposition')


@dataclass(frozen=True)
class Reconciliation(Metadata):
    schema_name: ClassVar[str] = 'reconciliation'
    schema_version: int
    scope: str
    source: str
    component: str | None
    label: str | None
    method: str | None
    language: str | None
    stage: str
    denominator: int | None
    count: int | None
    unknown_reason: str | None

    def __post_init__(self):
        _version(self.schema_version); _scope(self.scope)
        for name in ('source','stage'): _text(getattr(self,name),name)
        for name in ('component','label','method','language','unknown_reason'): _optional_text(getattr(self,name),name)
        _count(self.denominator,'denominator'); _count(self.count,'count')
        if (self.count is None or self.denominator is None) and not self.unknown_reason: raise ContractError('unknown counts require a reason')
        if self.count is not None and self.denominator is not None and self.count > self.denominator: raise ContractError('count exceeds denominator')


def validate_scope(rows, scope):
    """Prevent mixing fixture and research audit rows in one collection."""
    _scope(scope)
    if any(getattr(row, 'scope', None) != scope for row in rows):
        raise ContractError('mixed or unsupported audit row scope')
