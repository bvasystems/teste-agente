"""
Resultado de validação de guardrails.
"""

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class GuardrailAction(Enum):
    """Ações que podem ser tomadas baseadas no resultado do guardrail."""

    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    WARN = "warn"
    LOG = "log"


class GuardrailResult(BaseModel):
    """
    Resultado de uma validação de guardrail.

    Attributes:
        action: Ação recomendada (ALLOW, BLOCK, MODIFY, WARN, LOG)
        confidence: Pontuação de confiança (0.0 a 1.0)
        reason: Razão para a decisão
        validator_name: Nome do validador que gerou este resultado
        modified_content: Conteúdo modificado (quando action=MODIFY)
        metadata: Metadados adicionais específicos do validador
        processing_time_ms: Tempo de processamento em milissegundos
        timestamp: Timestamp da validação
    """

    action: GuardrailAction
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    validator_name: str
    modified_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def should_block(self) -> bool:
        """Verifica se o conteúdo deve ser bloqueado."""
        return self.action == GuardrailAction.BLOCK

    @property
    def should_modify(self) -> bool:
        """Verifica se o conteúdo deve ser modificado."""
        return self.action == GuardrailAction.MODIFY

    @property
    def is_violation(self) -> bool:
        """Verifica se foi detectada uma violação."""
        return self.action in [GuardrailAction.BLOCK, GuardrailAction.WARN]
