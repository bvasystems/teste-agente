"""
Decorador genérico para observabilidade que funciona com qualquer cliente OTel.

Este módulo fornece o decorador @observe que é completamente agnóstico ao provedor
de telemetria específico. Ele delega toda a lógica de tracing para os clientes
configurados na lista otel_clients do GenerationProvider.
"""

import functools
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Dict, List, TypeVar, cast, get_args

from rsb.coroutines.fire_and_forget import fire_and_forget

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.messages.message import Message
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tracing.otel_client import OtelClient

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
        decorator_start = time.perf_counter()
        logger.info(f"🔍 [TIMING] Starting @observe decorator for {func.__name__}")

        # Obter a instância do provider (self)
        provider_check_start = time.perf_counter()
        provider_self = args[0]

        # Verificar se é uma instância válida do GenerationProvider
        if not isinstance(provider_self, GenerationProvider):
            logger.warning(
                f"@observe decorator aplicado a método de classe não-GenerationProvider: {type(provider_self)}"
            )
            return await func(*args, **kwargs)

        provider_check_time = time.perf_counter() - provider_check_start
        logger.info(f"⏱️  [TIMING] Provider validation: {provider_check_time:.4f}s")

        # Obter lista de clientes OTel do provider
        clients_check_start = time.perf_counter()
        otel_clients: Sequence[OtelClient] = getattr(provider_self, "otel_clients", [])
        clients_check_time = time.perf_counter() - clients_check_start
        logger.info(
            f"⏱️  [TIMING] Getting OTel clients: {clients_check_time:.4f}s (found {len(otel_clients)} clients)"
        )

        # Se não há clientes configurados, executar função normalmente
        if not otel_clients:
            logger.debug("Nenhum cliente OTel configurado, executando sem tracing")
            function_start = time.perf_counter()
            result = await func(*args, **kwargs)
            function_time = time.perf_counter() - function_start
            total_time = time.perf_counter() - decorator_start
            logger.info(
                f"⏱️  [TIMING] Function execution (no tracing): {function_time:.4f}s"
            )
            logger.info(
                f"⏱️  [TIMING] Total decorator overhead (no tracing): {total_time - function_time:.4f}s"
            )
            return result

        # Extrair parâmetros da função
        params_extraction_start = time.perf_counter()
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
        params_extraction_time = time.perf_counter() - params_extraction_start
        logger.info(f"⏱️  [TIMING] Parameter extraction: {params_extraction_time:.4f}s")

        # Resolver model se for ModelKind
        model_resolution_start = time.perf_counter()
        model_kind_values = get_args(ModelKind)
        if model in model_kind_values:
            model_kind = cast(ModelKind, model)
            model = provider_self.map_model_kind_to_provider_model(model_kind)
        model_resolution_time = time.perf_counter() - model_resolution_start
        logger.info(
            f"⏱️  [TIMING] Model resolution: {model_resolution_time:.4f}s (resolved to: {model})"
        )

        # Preparar dados de entrada para tracing
        input_data_start = time.perf_counter()
        input_data = _prepare_input_data(
            messages=messages,
            response_schema=response_schema,
            tools=tools,
            generation_config=generation_config,
        )
        input_data_time = time.perf_counter() - input_data_start
        logger.info(f"⏱️  [TIMING] Input data preparation: {input_data_time:.4f}s")

        # Preparar metadados de trace
        metadata_start = time.perf_counter()
        trace_metadata = _prepare_trace_metadata(
            model=model,
            provider=provider_self.organization,
            generation_config=generation_config,
        )
        metadata_time = time.perf_counter() - metadata_start
        logger.info(f"⏱️  [TIMING] Trace metadata preparation: {metadata_time:.4f}s")

        # Extrair parâmetros de trace da configuração
        trace_params_start = time.perf_counter()
        trace_params = generation_config.trace_params
        user_id = trace_params.get("user_id")
        session_id = trace_params.get("session_id")
        tags = trace_params.get("tags")
        trace_params_time = time.perf_counter() - trace_params_start
        logger.info(f"⏱️  [TIMING] Trace params extraction: {trace_params_time:.4f}s")

        # Criar contextos de trace e geração para todos os clientes
        contexts_creation_start = time.perf_counter()
        active_contexts: List[Dict[str, Any]] = []

        for i, client in enumerate(otel_clients):
            client_start = time.perf_counter()
            try:
                # Criar contexto de trace
                trace_context_start = time.perf_counter()
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
                trace_context_time = time.perf_counter() - trace_context_start
                logger.info(
                    f"⏱️  [TIMING] Client {i + 1} trace context creation: {trace_context_time:.4f}s"
                )

                if trace_ctx:
                    # Criar contexto de geração
                    generation_context_start = time.perf_counter()
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
                    generation_context_time = (
                        time.perf_counter() - generation_context_start
                    )
                    logger.info(
                        f"⏱️  [TIMING] Client {i + 1} generation context creation: {generation_context_time:.4f}s"
                    )

                    active_contexts.append(
                        {
                            "client": client,
                            "trace_gen": trace_gen,
                            "trace_ctx": trace_ctx,
                            "generation_gen": generation_gen,
                            "generation_ctx": generation_ctx,
                        }
                    )

                client_time = time.perf_counter() - client_start
                logger.info(
                    f"⏱️  [TIMING] Client {i + 1} total setup: {client_time:.4f}s"
                )

            except Exception as e:
                client_error_time = time.perf_counter() - client_start
                logger.error(
                    f"Erro ao criar contextos de tracing para {type(client).__name__}: {e} (took {client_error_time:.4f}s)"
                )

        contexts_creation_time = time.perf_counter() - contexts_creation_start
        logger.info(
            f"⏱️  [TIMING] All contexts creation: {contexts_creation_time:.4f}s ({len(active_contexts)} active)"
        )

        # Registrar tempo de início
        start_time = datetime.now()
        setup_complete_time = time.perf_counter() - decorator_start
        logger.info(f"⏱️  [TIMING] Total setup time: {setup_complete_time:.4f}s")

        function_start = time.perf_counter()
        # Executar a função original
        logger.info(f"🚀 [TIMING] Starting main function execution: {func.__name__}")
        try:
            response = await func(*args, **kwargs)
            function_time = time.perf_counter() - function_start
            logger.info(f"✅ [TIMING] Main function completed: {function_time:.4f}s")

            # ✅ FIX: Processar resposta com sucesso de forma síncrona para dados críticos
            success_processing_start = time.perf_counter()
            await _process_successful_response(
                response=response,
                start_time=start_time,
                model=model,
                provider_self=provider_self,
                active_contexts=active_contexts,
                trace_metadata=trace_metadata,
            )
            success_processing_time = time.perf_counter() - success_processing_start
            logger.info(
                f"⏱️  [TIMING] Success response processing: {success_processing_time:.4f}s"
            )

            total_time = time.perf_counter() - decorator_start
            logger.info(
                f"🎯 [TIMING] Total @observe execution: {total_time:.4f}s (function: {function_time:.4f}s, overhead: {total_time - function_time:.4f}s)"
            )

            return response

        except Exception as e:
            function_error_time = (
                time.perf_counter() - function_start
                if "function_start" in locals()
                else 0
            )
            logger.error(
                f"❌ [TIMING] Function failed after {function_error_time:.4f}s: {e}"
            )

            # Processar erro
            error_processing_start = time.perf_counter()
            await _process_error_response(
                error=e,
                start_time=start_time,
                active_contexts=active_contexts,
                trace_metadata=trace_metadata,
            )
            error_processing_time = time.perf_counter() - error_processing_start
            logger.info(
                f"⏱️  [TIMING] Error response processing: {error_processing_time:.4f}s"
            )

            # Re-lançar a exceção para não alterar comportamento
            raise

        finally:
            # Limpar contextos
            cleanup_start = time.perf_counter()
            await _cleanup_contexts(active_contexts)
            cleanup_time = time.perf_counter() - cleanup_start
            logger.info(f"⏱️  [TIMING] Context cleanup: {cleanup_time:.4f}s")

    return cast(F, wrapper)


def _prepare_input_data(
    messages: List[Message],
    response_schema: Any,
    tools: Any,
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    """Prepara dados de entrada para tracing."""
    prep_start = time.perf_counter()

    messages_processing_start = time.perf_counter()
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
    messages_processing_time = time.perf_counter() - messages_processing_start
    logger.debug(
        f"⏱️  [TIMING] Messages processing in input_data: {messages_processing_time:.4f}s ({len(messages)} messages)"
    )

    # Adicionar parâmetros de configuração
    config_processing_start = time.perf_counter()
    if hasattr(generation_config, "__dict__"):
        for key, value in generation_config.__dict__.items():
            if (
                not key.startswith("_")
                and not callable(value)
                and key != "trace_params"
            ):
                input_data[key] = value
    config_processing_time = time.perf_counter() - config_processing_start
    logger.debug(
        f"⏱️  [TIMING] Config processing in input_data: {config_processing_time:.4f}s"
    )

    total_prep_time = time.perf_counter() - prep_start
    logger.debug(f"⏱️  [TIMING] Total input_data preparation: {total_prep_time:.4f}s")

    return input_data


def _prepare_trace_metadata(
    model: str,
    provider: str,
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    """Prepara metadados de trace."""
    metadata_start = time.perf_counter()

    trace_metadata = {
        "model": model,
        "provider": provider,
    }

    # Adicionar metadados customizados da configuração
    custom_metadata_start = time.perf_counter()
    trace_params = generation_config.trace_params
    if "metadata" in trace_params:
        metadata_val = trace_params["metadata"]
        if isinstance(metadata_val, dict):
            for k, v in metadata_val.items():
                if isinstance(k, str):
                    trace_metadata[k] = v
    custom_metadata_time = time.perf_counter() - custom_metadata_start
    logger.debug(f"⏱️  [TIMING] Custom metadata processing: {custom_metadata_time:.4f}s")

    total_metadata_time = time.perf_counter() - metadata_start
    logger.debug(
        f"⏱️  [TIMING] Total trace metadata preparation: {total_metadata_time:.4f}s"
    )

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
    processing_start = time.perf_counter()
    logger.info("📊 [TIMING] Starting successful response processing")

    # Extrair dados de uso
    usage_extraction_start = time.perf_counter()
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
    usage_extraction_time = time.perf_counter() - usage_extraction_start
    logger.info(f"⏱️  [TIMING] Usage extraction: {usage_extraction_time:.4f}s")

    # Calcular custos
    cost_calculation_start = time.perf_counter()
    cost_details = None
    if usage_details:
        input_tokens = int(usage_details.get("input", 0))
        output_tokens = int(usage_details.get("output", 0))

        if input_tokens > 0 or output_tokens > 0:
            try:
                pricing_start = time.perf_counter()
                input_cost = provider_self.price_per_million_tokens_input(
                    model, input_tokens
                ) * (input_tokens / 1_000_000)
                output_cost = provider_self.price_per_million_tokens_output(
                    model, output_tokens
                ) * (output_tokens / 1_000_000)
                total_cost = input_cost + output_cost
                pricing_time = time.perf_counter() - pricing_start
                logger.debug(f"⏱️  [TIMING] Pricing calculation: {pricing_time:.4f}s")

                if total_cost > 0:
                    cost_details = {
                        "input": round(input_cost, 8),
                        "output": round(output_cost, 8),
                        "total": round(total_cost, 8),
                        "currency": "USD",
                    }

                    logger.debug(
                        f"Calculated costs for {model}: total=${total_cost:.8f}"
                    )

            except Exception as e:
                logger.error(f"Error calculating costs: {e}")
    cost_calculation_time = time.perf_counter() - cost_calculation_start
    logger.info(f"⏱️  [TIMING] Cost calculation: {cost_calculation_time:.4f}s")

    # Preparar dados de saída
    output_data_start = time.perf_counter()
    output_data = {
        "completion": getattr(response, "text", str(response)),
    }
    output_data_time = time.perf_counter() - output_data_start
    logger.debug(f"⏱️  [TIMING] Output data preparation: {output_data_time:.4f}s")

    # ✅ FIX: Atualizar gerações de forma síncrona
    generation_updates_start = time.perf_counter()
    for i, ctx in enumerate(active_contexts):
        if ctx["generation_ctx"]:
            update_start = time.perf_counter()
            try:
                await ctx["client"].update_generation(
                    ctx["generation_ctx"],
                    output_data=output_data,
                    usage_details=usage_details,
                    cost_details=cost_details,
                    metadata=trace_metadata,
                )
                update_time = time.perf_counter() - update_start
                logger.debug(
                    f"⏱️  [TIMING] Generation update client {i + 1}: {update_time:.4f}s"
                )
            except Exception as e:
                update_time = time.perf_counter() - update_start
                logger.error(
                    f"Erro ao atualizar geração client {i + 1} (took {update_time:.4f}s): {e}"
                )
    generation_updates_time = time.perf_counter() - generation_updates_start
    logger.info(f"⏱️  [TIMING] All generation updates: {generation_updates_time:.4f}s")

    # ✅ FIX: Update trace with cost information for list view display
    trace_output_prep_start = time.perf_counter()
    parsed = getattr(response, "parsed", None)
    text = getattr(response, "text", str(response))
    final_output = parsed or text

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
            "currency": "USD",
        }

    # Add usage summary to trace output
    if usage_details:
        trace_output["usage_summary"] = {
            "total_tokens": usage_details["total"],
            "input_tokens": usage_details["input"],
            "output_tokens": usage_details["output"],
        }
    trace_output_prep_time = time.perf_counter() - trace_output_prep_start
    logger.debug(f"⏱️  [TIMING] Trace output preparation: {trace_output_prep_time:.4f}s")

    # Update traces
    trace_updates_start = time.perf_counter()
    for i, ctx in enumerate(active_contexts):
        if ctx["trace_ctx"]:
            trace_update_start = time.perf_counter()
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
                trace_update_time = time.perf_counter() - trace_update_start
                logger.debug(
                    f"⏱️  [TIMING] Trace update client {i + 1}: {trace_update_time:.4f}s"
                )
            except Exception as e:
                trace_update_time = time.perf_counter() - trace_update_start
                logger.error(
                    f"Error updating trace client {i + 1} (took {trace_update_time:.4f}s): {e}"
                )
    trace_updates_time = time.perf_counter() - trace_updates_start
    logger.info(f"⏱️  [TIMING] All trace updates: {trace_updates_time:.4f}s")

    # Continue with success scores (can be async)
    success_scores_start = time.perf_counter()
    for i, ctx in enumerate(active_contexts):
        if ctx["trace_ctx"]:
            fire_and_forget(
                _add_success_scores,
                ctx["client"],
                ctx["trace_ctx"],
                start_time,
                model,
                response,
            )
    success_scores_time = time.perf_counter() - success_scores_start
    logger.debug(
        f"⏱️  [TIMING] Success scores fire-and-forget dispatch: {success_scores_time:.4f}s"
    )

    total_processing_time = time.perf_counter() - processing_start
    logger.info(
        f"📊 [TIMING] Total successful response processing: {total_processing_time:.4f}s"
    )


async def _process_error_response(
    error: Exception,
    start_time: datetime,
    active_contexts: List[Dict[str, Any]],
    trace_metadata: Dict[str, Any],
) -> None:
    """Processa resposta com erro."""
    error_processing_start = time.perf_counter()
    logger.info("🚨 [TIMING] Starting error response processing")

    # Adicionar pontuações de erro
    error_scores_start = time.perf_counter()
    for _, ctx in enumerate(active_contexts):
        if ctx["trace_ctx"]:
            fire_and_forget(
                _add_error_scores, ctx["client"], ctx["trace_ctx"], error, start_time
            )
    error_scores_time = time.perf_counter() - error_scores_start
    logger.debug(
        f"⏱️  [TIMING] Error scores fire-and-forget dispatch: {error_scores_time:.4f}s"
    )

    # Tratar erro em todos os clientes
    error_handling_start = time.perf_counter()
    for _, ctx in enumerate(active_contexts):
        fire_and_forget(
            ctx["client"].handle_error,
            ctx["trace_ctx"],
            ctx["generation_ctx"],
            error,
            start_time,
            trace_metadata,
        )
    error_handling_time = time.perf_counter() - error_handling_start
    logger.debug(
        f"⏱️  [TIMING] Error handling fire-and-forget dispatch: {error_handling_time:.4f}s"
    )

    total_error_processing_time = time.perf_counter() - error_processing_start
    logger.info(
        f"🚨 [TIMING] Total error response processing: {total_error_processing_time:.4f}s"
    )


async def _cleanup_contexts(active_contexts: List[Dict[str, Any]]) -> None:
    """Limpa contextos de tracing."""
    cleanup_start = time.perf_counter()
    logger.info(
        f"🧹 [TIMING] Starting context cleanup for {len(active_contexts)} contexts"
    )

    for i, ctx in enumerate(active_contexts):
        client_cleanup_start = time.perf_counter()

        # Cleanup generation context
        gen_cleanup_start = time.perf_counter()
        try:
            if ctx["generation_gen"]:
                await ctx["generation_gen"].aclose()
        except Exception as e:
            logger.error(f"Erro ao fechar contexto de geração {i + 1}: {e}")
        gen_cleanup_time = time.perf_counter() - gen_cleanup_start
        logger.debug(
            f"⏱️  [TIMING] Generation context cleanup {i + 1}: {gen_cleanup_time:.4f}s"
        )

        # Cleanup trace context
        trace_cleanup_start = time.perf_counter()
        try:
            if ctx["trace_gen"]:
                await ctx["trace_gen"].aclose()
        except Exception as e:
            logger.error(f"Erro ao fechar contexto de trace {i + 1}: {e}")
        trace_cleanup_time = time.perf_counter() - trace_cleanup_start
        logger.debug(
            f"⏱️  [TIMING] Trace context cleanup {i + 1}: {trace_cleanup_time:.4f}s"
        )

        client_cleanup_time = time.perf_counter() - client_cleanup_start
        logger.debug(
            f"⏱️  [TIMING] Total client {i + 1} cleanup: {client_cleanup_time:.4f}s"
        )

    total_cleanup_time = time.perf_counter() - cleanup_start
    logger.info(f"🧹 [TIMING] Total context cleanup: {total_cleanup_time:.4f}s")


async def _add_success_scores(
    client: OtelClient,
    trace_ctx: Any,
    start_time: datetime,
    model: str,
    response: Generation[Any],
) -> None:
    """Adiciona pontuações de sucesso ao trace."""
    scores_start = time.perf_counter()
    logger.debug("🏆 [TIMING] Starting success scores addition")

    try:
        # Pontuação principal de sucesso
        success_score_start = time.perf_counter()
        await client.add_trace_score(
            trace_ctx,
            name="trace_success",
            value=1.0,
            comment="Generation completed successfully",
        )
        success_score_time = time.perf_counter() - success_score_start
        logger.debug(f"⏱️  [TIMING] Success score: {success_score_time:.4f}s")

        # Pontuação de latência
        latency_score_start = time.perf_counter()
        latency_seconds = (datetime.now() - start_time).total_seconds()
        latency_score = _calculate_latency_score(latency_seconds)
        await client.add_trace_score(
            trace_ctx,
            name="latency_score",
            value=latency_score,
            comment=f"Response time: {latency_seconds:.2f}s",
        )
        latency_score_time = time.perf_counter() - latency_score_start
        logger.debug(f"⏱️  [TIMING] Latency score: {latency_score_time:.4f}s")

        # Pontuação de tier do modelo
        model_tier_start = time.perf_counter()
        model_tier = _calculate_model_tier_score(model)
        await client.add_trace_score(
            trace_ctx,
            name="model_tier",
            value=model_tier,
            comment=f"Model capability tier: {model}",
        )
        model_tier_time = time.perf_counter() - model_tier_start
        logger.debug(f"⏱️  [TIMING] Model tier score: {model_tier_time:.4f}s")

        # Pontuação de uso de ferramentas
        tool_usage_start = time.perf_counter()
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
        tool_usage_time = time.perf_counter() - tool_usage_start
        logger.debug(f"⏱️  [TIMING] Tool usage score: {tool_usage_time:.4f}s")

        total_scores_time = time.perf_counter() - scores_start
        logger.debug(f"🏆 [TIMING] Total success scores: {total_scores_time:.4f}s")

    except Exception as e:
        scores_error_time = time.perf_counter() - scores_start
        logger.error(
            f"Erro ao adicionar pontuações de sucesso (took {scores_error_time:.4f}s): {e}"
        )


async def _add_error_scores(
    client: OtelClient,
    trace_ctx: Any,
    error: Exception,
    start_time: datetime,
) -> None:
    """Adiciona pontuações de erro ao trace."""
    error_scores_start = time.perf_counter()
    logger.debug("💥 [TIMING] Starting error scores addition")

    try:
        error_type = type(error).__name__
        error_str = str(error)

        # Pontuação principal de falha
        failure_score_start = time.perf_counter()
        await client.add_trace_score(
            trace_ctx,
            name="trace_success",
            value=0.0,
            comment=f"Error: {error_type} - {error_str[:100]}",
        )
        failure_score_time = time.perf_counter() - failure_score_start
        logger.debug(f"⏱️  [TIMING] Failure score: {failure_score_time:.4f}s")

        # Categoria do erro
        error_category_start = time.perf_counter()
        error_category = _categorize_error(error)
        await client.add_trace_score(
            trace_ctx,
            name="error_category",
            value=error_category,
            comment=f"Error classified as: {error_category}",
        )
        error_category_time = time.perf_counter() - error_category_start
        logger.debug(f"⏱️  [TIMING] Error category score: {error_category_time:.4f}s")

        # Severidade do erro
        severity_start = time.perf_counter()
        severity = _calculate_error_severity(error_category)
        await client.add_trace_score(
            trace_ctx,
            name="error_severity",
            value=severity,
            comment=f"Error severity: {severity:.1f}",
        )
        severity_time = time.perf_counter() - severity_start
        logger.debug(f"⏱️  [TIMING] Error severity score: {severity_time:.4f}s")

        # Latência até erro
        error_latency_start = time.perf_counter()
        error_latency = (datetime.now() - start_time).total_seconds()
        await client.add_trace_score(
            trace_ctx,
            name="error_latency",
            value=error_latency,
            comment=f"Time until error: {error_latency:.2f}s",
        )
        error_latency_time = time.perf_counter() - error_latency_start
        logger.debug(f"⏱️  [TIMING] Error latency score: {error_latency_time:.4f}s")

        total_error_scores_time = time.perf_counter() - error_scores_start
        logger.debug(f"💥 [TIMING] Total error scores: {total_error_scores_time:.4f}s")

    except Exception as e:
        error_scores_error_time = time.perf_counter() - error_scores_start
        logger.error(
            f"Erro ao adicionar pontuações de erro (took {error_scores_error_time:.4f}s): {e}"
        )


def _calculate_latency_score(latency_seconds: float) -> float:
    """Calcula pontuação baseada na latência."""
    calculation_start = time.perf_counter()

    if latency_seconds < 1.0:
        score = 1.0  # Excelente (sub-segundo)
    elif latency_seconds < 3.0:
        score = 0.8  # Bom (1-3 segundos)
    elif latency_seconds < 6.0:
        score = 0.6  # Aceitável (3-6 segundos)
    elif latency_seconds < 10.0:
        score = 0.4  # Lento (6-10 segundos)
    else:
        score = 0.2  # Muito lento (>10 segundos)

    calculation_time = time.perf_counter() - calculation_start
    logger.debug(f"⏱️  [TIMING] Latency score calculation: {calculation_time:.6f}s")
    return score


def _calculate_model_tier_score(model: str) -> float:
    """Calcula pontuação baseada no tier do modelo."""
    calculation_start = time.perf_counter()

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
        score = 1.0
    elif any(
        mid in model_name
        for mid in ["gemini-1.5-flash", "gemini-2.5-flash", "claude-3-haiku", "gpt-3.5"]
    ):
        score = 0.7
    else:
        score = 0.5  # Modelos básicos

    calculation_time = time.perf_counter() - calculation_start
    logger.debug(f"⏱️  [TIMING] Model tier score calculation: {calculation_time:.6f}s")
    return score


def _categorize_error(error: Exception) -> str:
    """Categoriza o tipo de erro."""
    categorization_start = time.perf_counter()

    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    if "timeout" in error_str or "time" in error_type:
        category = "timeout"
    elif "connection" in error_str or "network" in error_str:
        category = "network"
    elif "auth" in error_str or "key" in error_str or "credential" in error_str:
        category = "authentication"
    elif "limit" in error_str or "quota" in error_str or "rate" in error_str:
        category = "rate_limit"
    elif "value" in error_type or "type" in error_type or "attribute" in error_type:
        category = "validation"
    elif "memory" in error_str or "resource" in error_str:
        category = "resource"
    else:
        category = "other"

    categorization_time = time.perf_counter() - categorization_start
    logger.debug(
        f"⏱️  [TIMING] Error categorization: {categorization_time:.6f}s (category: {category})"
    )
    return category


def _calculate_error_severity(error_category: str) -> float:
    """Calcula severidade do erro baseada na categoria."""
    severity_start = time.perf_counter()

    if error_category in ["timeout", "network", "rate_limit"]:
        severity = 0.5  # Erros transitórios - menor severidade
    elif error_category in ["authentication", "validation"]:
        severity = 0.9  # Erros de configuração/código - maior severidade
    else:
        severity = 0.7  # Severidade média-alta por padrão

    severity_time = time.perf_counter() - severity_start
    logger.debug(
        f"⏱️  [TIMING] Error severity calculation: {severity_time:.6f}s (severity: {severity})"
    )
    return severity
