An inference gateway sits between applications and large language model providers. It centralizes authentication, request logging, per-team quotas, and provider routing behind one HTTP endpoint.

When a provider answers with HTTP 429 or a 5xx error, the gateway can retry with exponential backoff, honor the Retry-After header, and fall back to a secondary provider in a configured chain.

Timeouts should be set per request. A slow upstream model call that exceeds its deadline is cancelled so the caller gets a clear error instead of hanging indefinitely.

Gateways often cache identical prompts. An exact-match cache keyed on the model name and normalized prompt returns a stored completion without calling the provider again.
