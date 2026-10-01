# RAGOps Python SDK

Instrument a RAG or LLM application so it reports its own retrieval, generation,
token and latency numbers to a [RAGOps](https://github.com/MoAftab-01/RAGOps)
server.

The SDK does three things and nothing else: record what happened, batch it in the
background, and ship it. It computes no metrics and calls no model. Token
counts are a string operation; faithfulness scoring is the server's job.

## Install

```bash
pip install ragops-sdk
```

From a checkout:

```bash
cd sdk/python && pip install -e .
```

## Configure

```python
from ragops import RAGOps

client = RAGOps(
    endpoint="http://localhost:8000",
    api_key="dev-key",              # per-company keys work too
    application="customer-support-bot",
)
```

`endpoint` defaults to `http://localhost:8000`. There is no environment
variable read and no config file — pass `api_key` explicitly, or read it
yourself and hand it in. `close()` is registered with `atexit`, so events
buffered at interpreter shutdown are flushed on the way out.

## Record a trace

`client.trace(...)` is a context manager. On a clean exit the trace completes
with status `success` and flushes. If the block raises, the error is recorded,
the status becomes `error`, and the exception propagates unchanged — instrumenting
never swallows your failures.

```python
def answer(question: str) -> str:
    with client.trace(name="answer_question") as trace:
        docs = retriever.search(question, k=5)
        trace.log_retrieval(
            query=question,
            documents=[{"id": d.id, "text": d.text, "score": d.score} for d in docs],
            top_k=5,
        )

        prompt = build_prompt(question, docs)
        answer = llm.generate(prompt)

        trace.log_generation(
            model="llama3.1",
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(answer),
            answer=answer,
        )
        trace.set_output(answer)

    return answer
```

`estimate_tokens` is the SDK's own counter, `from ragops.tokens import
estimate_tokens`. Count however your system already counts — the only rule is
that the number is measured rather than guessed by a model.

### Method reference

| Method | What to pass |
|---|---|
| `trace.log_retrieval(query, documents, retriever, top_k, duration_ms, configuration)` | The query you searched for and what came back. `top_k`, not `k`. |
| `trace.log_generation(model, answer, input_tokens, output_tokens, prompt, duration_ms, provider, temperature, max_tokens, time_to_first_token_ms, status, metadata, span_id)` | One model call. `answer`/`input_tokens`/`output_tokens`, not `output_text`/`prompt_tokens`/`completion_tokens`. |
| `trace.log_span(name, start_time, end_time, attributes)` | Any timed step that is not a retrieval or a generation. |
| `trace.log_agent_step(step, name, ...)` | One step of a multi-step agent. |
| `trace.set_error(exc)` / `trace.set_tags([...])` / `trace.set_metadata({...})` | Annotate without ending the trace. |
| `trace.set_output(text)` | The final answer, for the trace list view. |

None of these accept arbitrary keyword arguments. An unknown name raises
`TypeError` at the call site rather than being silently dropped, which is the
point: a typo in a metric name should fail loudly rather than record nothing.

## Batching

Events are buffered and flushed on a background timer, so instrumentation adds
no latency to the traced call. `flush_interval` (default 5.0s) is the timer and
`batch_size` (default 50) triggers an immediate flush once that many items are
pending.

The queue is capped. Past `10 * batch_size` pending items the oldest are dropped
with a warning rather than letting an unreachable server grow the buffer without
bound — telemetry must never become an outage. `client.flush()` forces a drain
now, and `client.close()` shuts the batcher down and waits for in-flight work.
`health()` returns the server's status dict, or `None` if it cannot be reached.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT. See [LICENSE](../../LICENSE).