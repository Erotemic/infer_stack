# Benchmark candidate index

Routing table for benchmark questions. When working in an area below,
read the linked entries before designing the change.

## By invariant tag

| Tag | Entries | Failure class |
|---|---|---|
| `lifecycle-ordering` | [Q001](questions.md#q001--docker-compose-dependson-cascades-recreates-into-dependents) | Long-lived dependents bounced when a depended-on service is recreated |
| `dependency-boundary` | [Q001](questions.md#q001--docker-compose-dependson-cascades-recreates-into-dependents) | `depends_on` chosen as ordering hint becomes a load-bearing recreate trigger |

## By surface

| Surface | Entries |
|---|---|
| Docker Compose | Q001 |
