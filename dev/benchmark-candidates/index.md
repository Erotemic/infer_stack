# Benchmark candidate index

Routing table for benchmark questions. When working in an area below,
read the linked entries before designing the change.

## By invariant tag

| Tag | Entries | Failure class |
|---|---|---|
| `lifecycle-ordering` | [Q001](questions.md#q001--docker-compose-dependson-cascades-recreates-into-dependents) | Long-lived dependents bounced when a depended-on service is recreated |
| `dependency-boundary` | [Q001](questions.md#q001--docker-compose-dependson-cascades-recreates-into-dependents) | `depends_on` chosen as ordering hint becomes a load-bearing recreate trigger |
| `xdg-basedir` | [Q002](questions.md#q002--xdg-basedir-pick-data-for-persistent-state-not-cache) | Defaulting persistent user data into `XDG_CACHE_HOME` (silently wipeable) |
| `persistence` | [Q002](questions.md#q002--xdg-basedir-pick-data-for-persistent-state-not-cache) | Regenerable vs non-regenerable directories conflated under one root |
| `pattern-following` | [Q002](questions.md#q002--xdg-basedir-pick-data-for-persistent-state-not-cache) | Copying a sibling's idiom when the new use has different semantics |

## By surface

| Surface | Entries |
|---|---|
| Docker Compose | Q001 |
| Filesystem layout / XDG | Q002 |
