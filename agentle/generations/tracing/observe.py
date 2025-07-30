"""
Decorador genérico para observabilidade que funciona com qualquer cliente OTel.

Este módulo fornece o decorador @observe que é completamente agnóstico ao provedor
de telemetria específico. Ele delega toda a lógica de tracing para os clientes
configurados na lista otel_clients do GenerationProvider.
"""

import functools
import inspect
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, TypeVar, cast, get_args, Dict, List

from rsb.coroutines.fire_and_forget import fire_and_forget

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.models.messages.message import Message
from .otel_client import OtelClient

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def observe(func: F) -> F:
    """
    Decorador genérico para observabilidade que funciona com qualquer cliente OTel.

    Este decorador é completamente agnóstico ao provedor de telemetria específico.
    Ele obtém a lista de clientes da instância do provider e delega toda a lógica
    de tracing para cada cliente configurado.

    Características principais:
    - Agnóstico ao provedor: funciona com qualquer implementação de OtelClient
    - Tratamento robusto de erros: falhas de telemetria não interrompem execução
    - Suporte a múltiplos clientes: pode enviar dados para várias destinações
    - Performance otimizada: operações de telemetria são não-bloqueantes
    - Coleta automática de métricas: tokens, custos, latência, etc.

    Usage:
        ```python
        class MyProvider(GenerationProvider):
            @observe
            async def generate_async(self, ...) -> Generation[T]:
                # Lógica de geração aqui
                return generation
        ```

    Args:
        func: O método de geração a ser decorado

    Returns:
        Função decorada com observabilidade automática
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Generation[Any]:
        # Obter a instância do provider (self)
        provider_self = args[0]

        # Verificar se é uma instância válida do GenerationProvider
        if not isinstance(provider_self, GenerationProvider):
            logger.warning(
                f"@observe decorator aplicado a método de classe não-GenerationProvider: {type(provider_self)}"
            )
            return await func(*args, **kwargs)

        # Obter lista de clientes OTel do provider
        otel_clients: Sequence[OtelClient] = getattr(provider_self, "otel_clients", [])

        # Se não há clientes configurados, executar função normalmente
        if not otel_clients:
            logger.debug("Nenhum cliente OTel configurado, executando sem tracing")
            return await func(*args, **kwargs)

        # Extrair parâmetros da função
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extrair parâmetros relevantes para tracing
        model = bound_args.arguments.get("model") or provider_self.default_model
        messages = bound_args.arguments.get("messages", [])
        response_schema = bound_args.arguments.get("response_schema")
        generation_config = (
            bound_args.arguments.get("generation_config") or GenerationConfig()
        )
        tools = bound_args.arguments.get("tools")

        # Resolver model se for ModelKind
        model_kind_values = get_args(ModelKind)
        if model in model_kind_values:
            model_kind = cast(ModelKind, model)
            model = provider_self.map_model_kind_to_provider_model(model_kind)

        # Preparar dados de entrada para tracing
        input_data = _prepare_input_data(
            messages=messages,
            response_schema=response_schema,
            tools=tools,
            generation_config=generation_config,
        )

        # Preparar metadados de trace
        trace_metadata = _prepare_trace_metadata(
            model=model,
            provider=provider_self.organization,
            generation_config=generation_config,
        )

        # Extrair parâmetros de trace da configuração
        trace_params = generation_config.trace_params
        user_id = trace_params.get("user_id")
        session_id = trace_params.get("session_id")
        tags = trace_params.get("tags")

        # Criar contextos de trace e geração para todos os clientes
        active_contexts: List[Dict[str, Any]] = []

        for client in otel_clients:
            try:
                # Criar contexto de trace
                trace_gen = client.trace_context(
                    name=trace_params.get(
                        "name", f"{provider_self.organization}_{model}_conversation"
                    ),
                    input_data=input_data,
                    metadata=trace_metadata,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                )
                trace_ctx = await trace_gen.__anext__()

                if trace_ctx:
                    # Criar contexto de geração
                    generation_gen = client.generation_context(
                        trace_context=trace_ctx,
                        name=trace_params.get(
                            "name", f"{provider_self.organization}_{model}_generation"
                        ),
                        model=model,
                        provider=provider_self.organization,
                        input_data=input_data,
                        metadata=trace_metadata,
                    )
                    generation_ctx = await generation_gen.__anext__()

                    active_contexts.append(
                        {
                            "client": client,
                            "trace_gen": trace_gen,
                            "trace_ctx": trace_ctx,
                            "generation_gen": generation_gen,
                            "generation_ctx": generation_ctx,
                        }
                    )

            except Exception as e:
                logger.error(
                    f"Erro ao criar contextos de tracing para {type(client).__name__}: {e}"
                )

        # Registrar tempo de início
        start_time = datetime.now()

        try:
            # Executar a função original
            response = await func(*args, **kwargs)

            # ✅ FIX: Processar resposta com sucesso de forma síncrona para dados críticos
            await _process_successful_response(
                response=response,
                start_time=start_time,
                model=model,
                provider_self=provider_self,
                active_contexts=active_contexts,
                trace_metadata=trace_metadata,
            )

            return response

        except Exception as e:
            # Processar erro
            await _process_error_response(
                error=e,
                start_time=start_time,
                active_contexts=active_contexts,
                trace_metadata=trace_metadata,
            )

            # Re-lançar a exceção para não alterar comportamento
            raise

        finally:
            # Limpar contextos
            await _cleanup_contexts(active_contexts)

    return cast(F, wrapper)


def _prepare_input_data(
    messages: List[Message],
    response_schema: Any,
    tools: Any,
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    """Prepara dados de entrada para tracing."""
    input_data = {
        "messages": [
            {
                "role": msg.role,
                "content": "".join(str(part) for part in msg.parts),
            }
            for msg in messages
        ],
        "response_schema": str(response_schema) if response_schema else None,
        "tools_count": len(tools) if tools else 0,
        "message_count": len(messages),
        "has_tools": tools is not None and len(tools) > 0,
        "has_schema": response_schema is not None,
    }

    # Adicionar parâmetros de configuração
    if hasattr(generation_config, "__dict__"):
        for key, value in generation_config.__dict__.items():
            if (
                not key.startswith("_")
                and not callable(value)
                and key != "trace_params"
            ):
                input_data[key] = value

    return input_data


def _prepare_trace_metadata(
    model: str,
    provider: str,
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    """Prepara metadados de trace."""
    trace_metadata = {
        "model": model,
        "provider": provider,
    }

    # Adicionar metadados customizados da configuração
    trace_params = generation_config.trace_params
    if "metadata" in trace_params:
        metadata_val = trace_params["metadata"]
        if isinstance(metadata_val, dict):
            for k, v in metadata_val.items():
                if isinstance(k, str):
                    trace_metadata[k] = v

    return trace_metadata


async def _process_successful_response(
    response: Generation[Any],
    start_time: datetime,
    model: str,
    provider_self: GenerationProvider,
    active_contexts: List[Dict[str, Any]],
    trace_metadata: Dict[str, Any],
) -> None:
    """Processa resposta bem-sucedida."""
    # Extrair dados de uso
    usage_details = None
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

        usage_details = {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": total_tokens,
            "unit": "TOKENS",
        }

    # Calcular custos
    cost_details = None
    if usage_details:
        input_tokens = int(usage_details.get("input", 0))
        output_tokens = int(usage_details.get("output", 0))

        if input_tokens > 0 or output_tokens > 0:
            try:
                input_cost = provider_self.price_per_million_tokens_input(
                    model, input_tokens
                ) * (input_tokens / 1_000_000)
                output_cost = provider_self.price_per_million_tokens_output(
                    model, output_tokens
                ) * (output_tokens / 1_000_000)
                total_cost = input_cost + output_cost

                if total_cost > 0:
                    cost_details = {
                        "input": round(input_cost, 8),
                        "output": round(output_cost, 8),
                        "total": round(total_cost, 8),
                        "currency": "USD"
                    }
                    
                    logger.debug(f"Calculated costs for {model}: total=${total_cost:.8f}")
                    
            except Exception as e:
                logger.error(f"Error calculating costs: {e}")

    # Preparar dados de saída
    output_data = {
        "completion": getattr(response, "text", str(response)),
    }

    # ✅ FIX: Atualizar gerações de forma síncrona
    for ctx in active_contexts:
        if ctx["generation_ctx"]:
            try:
                await ctx["client"].update_generation(
                    ctx["generation_ctx"],
                    output_data=output_data,
                    usage_details=usage_details,
                    cost_details=cost_details,
                    metadata=trace_metadata,
                )
            except Exception as e:
                logger.error(f"Erro ao atualizar geração: {e}")

    # ✅ FIX: Update trace with cost information for list view display
    parsed = getattr(response, "parsed", None)
    text = getattr(response, "text", str(response))
    final_output = parsed or text

    for ctx in active_contexts:
        if ctx["trace_ctx"]:
            # Prepare trace output with cost summary
            trace_output = {
                "result": final_output,
            }

            # Add cost summary to trace output if available
            if cost_details:
                trace_output["cost_summary"] = {
                    "total_cost": cost_details["total"],
                    "input_cost": cost_details["input"], 
                    "output_cost": cost_details["output"],
                    "currency": "USD"
                }
            
            # Add usage summary to trace output
            if usage_details:
                trace_output["usage_summary"] = {
                    "total_tokens": usage_details["total"],
                    "input_tokens": usage_details["input"],
                    "output_tokens": usage_details["output"]
                }
            
            try:
                await ctx["client"].update_trace(
                    ctx["trace_ctx"],
                    output_data=trace_output,
                    success=True,
                    metadata={
                        **trace_metadata,
                        # ✅ Add cost metadata at trace level
                        "total_cost": cost_details["total"] if cost_details else 0.0,
                        "cost_currency": "USD",
                        "total_tokens": usage_details["total"] if usage_details else 0,
                    },
                )
            except Exception as e:
                logger.error(f"Error updating trace: {e}")

    # Continue with success scores (can be async)
    for ctx in active_contexts:
        if ctx["trace_ctx"]:
            fire_and_forget(
                _add_success_scores,
                ctx["client"],
                ctx["trace_ctx"],
                start_time,
                model,
                response,
            )


async def _process_error_response(
    error: Exception,
    start_time: datetime,
    active_contexts: List[Dict[str, Any]],
    trace_metadata: Dict[str, Any],
) -> None:
    """Processa resposta com erro."""
    # Adicionar pontuações de erro
    for ctx in active_contexts:
        if ctx["trace_ctx"]:
            fire_and_forget(
                _add_error_scores, ctx["client"], ctx["trace_ctx"], error, start_time
            )

    # Tratar erro em todos os clientes
    for ctx in active_contexts:
        fire_and_forget(
            ctx["client"].handle_error,
            ctx["trace_ctx"],
            ctx["generation_ctx"],
            error,
            start_time,
            trace_metadata,
        )


async def _cleanup_contexts(active_contexts: List[Dict[str, Any]]) -> None:
    """Limpa contextos de tracing."""
    for ctx in active_contexts:
        try:
            if ctx["generation_gen"]:
                await ctx["generation_gen"].aclose()
        except Exception as e:
            logger.error(f"Erro ao fechar contexto de geração: {e}")

        try:
            if ctx["trace_gen"]:
                await ctx["trace_gen"].aclose()
        except Exception as e:
            logger.error(f"Erro ao fechar contexto de trace: {e}")


async def _add_success_scores(
    client: OtelClient,
    trace_ctx: Any,
    start_time: datetime,
    model: str,
    response: Generation[Any],
) -> None:
    """Adiciona pontuações de sucesso ao trace."""
    try:
        # Pontuação principal de sucesso
        await client.add_trace_score(
            trace_ctx,
            name="trace_success",
            value=1.0,
            comment="Generation completed successfully",
        )

        # Pontuação de latência
        latency_seconds = (datetime.now() - start_time).total_seconds()
        latency_score = _calculate_latency_score(latency_seconds)
        await client.add_trace_score(
            trace_ctx,
            name="latency_score",
            value=latency_score,
            comment=f"Response time: {latency_seconds:.2f}s",
        )

        # Pontuação de tier do modelo
        model_tier = _calculate_model_tier_score(model)
        await client.add_trace_score(
            trace_ctx,
            name="model_tier",
            value=model_tier,
            comment=f"Model capability tier: {model}",
        )

        # Pontuação de uso de ferramentas
        tool_calls = response.tool_calls
        if hasattr(response, "tools") or len(tool_calls) > 0:
            tool_usage_score = 1.0 if tool_calls and len(tool_calls) > 0 else 0.0
            tool_comment = (
                f"Tools were used ({len(tool_calls)} function calls)"
                if tool_usage_score > 0
                else "Tools were available but not used"
            )
            await client.add_trace_score(
                trace_ctx,
                name="tool_usage",
                value=tool_usage_score,
                comment=tool_comment,
            )

    except Exception as e:
        logger.error(f"Erro ao adicionar pontuações de sucesso: {e}")


async def _add_error_scores(
    client: OtelClient,
    trace_ctx: Any,
    error: Exception,
    start_time: datetime,
) -> None:
    """Adiciona pontuações de erro ao trace."""
    try:
        error_type = type(error).__name__
        error_str = str(error)

        # Pontuação principal de falha
        await client.add_trace_score(
            trace_ctx,
            name="trace_success",
            value=0.0,
            comment=f"Error: {error_type} - {error_str[:100]}",
        )

        # Categoria do erro
        error_category = _categorize_error(error)
        await client.add_trace_score(
            trace_ctx,
            name="error_category",
            value=error_category,
            comment=f"Error classified as: {error_category}",
        )

        # Severidade do erro
        severity = _calculate_error_severity(error_category)
        await client.add_trace_score(
            trace_ctx,
            name="error_severity",
            value=severity,
            comment=f"Error severity: {severity:.1f}",
        )

        # Latência até erro
        error_latency = (datetime.now() - start_time).total_seconds()
        await client.add_trace_score(
            trace_ctx,
            name="error_latency",
            value=error_latency,
            comment=f"Time until error: {error_latency:.2f}s",
        )

    except Exception as e:
        logger.error(f"Erro ao adicionar pontuações de erro: {e}")


def _calculate_latency_score(latency_seconds: float) -> float:
    """Calcula pontuação baseada na latência."""
    if latency_seconds < 1.0:
        return 1.0  # Excelente (sub-segundo)
    elif latency_seconds < 3.0:
        return 0.8  # Bom (1-3 segundos)
    elif latency_seconds < 6.0:
        return 0.6  # Aceitável (3-6 segundos)
    elif latency_seconds < 10.0:
        return 0.4  # Lento (6-10 segundos)
    else:
        return 0.2  # Muito lento (>10 segundos)


def _calculate_model_tier_score(model: str) -> float:
    """Calcula pontuação baseada no tier do modelo."""
    model_name = model.lower()

    # Modelos avançados recebem pontuação alta
    if any(
        premium in model_name
        for premium in [
            "gpt-4",
            "claude-3-opus",
            "claude-3-sonnet",
            "gemini-1.5-pro",
            "gemini-2.0-pro",
            "claude-3-7",
        ]
    ):
        return 1.0
    elif any(
        mid in model_name
        for mid in ["gemini-1.5-flash", "gemini-2.5-flash", "claude-3-haiku", "gpt-3.5"]
    ):
        return 0.7
    else:
        return 0.5  # Modelos básicos


def _categorize_error(error: Exception) -> str:
    """Categoriza o tipo de erro."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    if "timeout" in error_str or "time" in error_type:
        return "timeout"
    elif "connection" in error_str or "network" in error_str:
        return "network"
    elif "auth" in error_str or "key" in error_str or "credential" in error_str:
        return "authentication"
    elif "limit" in error_str or "quota" in error_str or "rate" in error_str:
        return "rate_limit"
    elif "value" in error_type or "type" in error_type or "attribute" in error_type:
        return "validation"
    elif "memory" in error_str or "resource" in error_str:
        return "resource"
    else:
        return "other"


def _calculate_error_severity(error_category: str) -> float:
    """Calcula severidade do erro baseada na categoria."""
    if error_category in ["timeout", "network", "rate_limit"]:
        return 0.5  # Erros transitórios - menor severidade
    elif error_category in ["authentication", "validation"]:
        return 0.9  # Erros de configuração/código - maior severidade
    else:
        return 0.7  # Severidade média-alta por padrão
