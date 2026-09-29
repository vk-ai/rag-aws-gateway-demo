Amazon Bedrock exposes hosted foundation models through a single application programming interface. A learning demo can keep a stub provider as the default and only call Bedrock when credentials and an explicit opt-in flag are set.

IAM policies control which principals may invoke which models. A least-privilege policy grants bedrock:InvokeModel only on the specific model identifiers the service actually uses.

Amazon S3 stores the raw documents for a retrieval corpus. An ingest job lists new objects, splits them into chunks, embeds each chunk, and writes the vectors to an index.

CloudWatch collects latency and error metrics for the gateway. Alarms on p95 latency and on the 5xx rate warn operators before users notice slow or failed answers.
