# Future Integration: ternary-fitness

## Current State
Fitness landscape analysis with `Entropy`, `Environment`, `FitnessEvaluator`, `ExhaustiveSearch`, `FitnessLandscape` with `LandscapePoint` and `SaddlePoint`, `ParetoFront` with `Objective` and `ParetoSolution`, and `TernaryStrategy`. Analyzes strategy fitness over {-1, 0, +1} action spaces.

## Integration Opportunities

### With strategy-ecology
`FitnessLandscape` IS the ecology. `TernaryStrategy` maps to strategy-ecology's strategy species. `LandscapePoint` with its fitness value determines species survival. `SaddlePoint` identifies evolutionary transitions — where one strategy species outcompetes another. `ParetoFront` with multiple `Objective` dimensions becomes the multi-niche ecosystem where different species coexist by optimizing different objectives.

### With ternary-agent
`AgentPool::rank_by_fitness()` uses `FitnessEvaluator` to compute actual fitness. Agents are placed on the `FitnessLandscape` based on their strategy. The landscape's topology guides agent exploration — agents move toward local fitness peaks, and `SaddlePoint` detection identifies when an agent should leap to a higher peak.

### With ternary-compiler
`CompiledPolicy` strategies are evaluated by `FitnessEvaluator`. `ExhaustiveSearch` finds optimal strategies, which are then compiled by `ternary-compiler` for deployment. The feedback loop: compile → evaluate → optimize → recompile.

## Potential in Mature Systems
Fitness landscape becomes the universal optimization framework. Fleet resource allocation is a fitness landscape. Agent task assignment is a fitness landscape. Room topology optimization is a fitness landscape. `ParetoFront` handles multi-objective decisions everywhere — speed vs. cost, accuracy vs. latency, exploration vs. exploitation.

## Cross-Pollination Ideas
- `Entropy` module connects directly to `ternary-entropy` for Shannon entropy of strategy distributions
- `ParetoFront` could be visualized in `ternary-spreadsheet` with Pareto-optimal cells highlighted
- `SaddlePoint` detection connects to `ternary-inference` — saddle points are gaps in the fitness landscape
- `ExhaustiveSearch` results feed into `ternary-science` for experimental validation

## Dependencies for Next Steps
- GPU acceleration for `ExhaustiveSearch` on large strategy spaces (connect to ternary-science GPU benchmarks)
- Streaming fitness evaluation for live agent populations
- Integration with ternary-compiler for strategy→policy→evaluation pipeline
- Multi-objective extension connecting to strategy-ecology's niche model
