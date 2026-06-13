# Ternary Fitness — Landscape Analysis for Ternary Agent Strategy Evolution

**Ternary Fitness** analyzes the fitness landscape of ternary strategy spaces — the set of all possible strategies where each decision dimension takes a value in {-1, 0, +1}. It provides exhaustive landscape enumeration, topology analysis (peaks, valleys, saddles), Pareto front computation for multi-objective optimization, and entropy measures of the strategy distribution.

## Why It Matters

Evolutionary optimization is only as good as the landscape it explores. If the landscape is smooth with a single peak, any hill-climbing algorithm works. If it's rugged with many local optima, the search strategy matters enormously. For ternary agents, the landscape is discrete: each strategy is a vector of trits, and the landscape size is 3ⁿ for n decision dimensions. This crate exhaustively characterizes these landscapes — finding all peaks, detecting saddle points, measuring ruggedness — enabling informed choices about which optimization algorithm to use and which strategies are evolutionarily stable.

## How It Works

### Exhaustive Search

For small strategy spaces (n ≤ 15, giving ≤ 14,348,907 strategies), the `ExhaustiveSearch` evaluator computes fitness for every possible ternary strategy. This gives the complete landscape — no sampling bias, no missed peaks. The evaluator runs in O(3ⁿ) time.

### Landscape Topology

The `FitnessLandscape` analysis identifies:

- **Peaks**: Strategies with higher fitness than all 2n Hamming-1 neighbors
- **Valleys**: Strategies with lower fitness than all neighbors
- **Saddle points**: Strategies that are local maxima in some directions but not all
- **Plateaus**: Connected regions of equal fitness

The number of peaks measures landscape ruggedness: more peaks = harder optimization.

### Pareto Fronts

For multi-objective fitness (e.g., accuracy vs. efficiency), the `ParetoFront` identifies non-dominated strategies — those where no other strategy is better on all objectives. The Pareto front size measures how many trade-off options exist.

### Entropy

Strategy distribution entropy:

```
H = -Σ p(s) · log p(s)   over strategies s
```

High entropy means strategies are uniformly distributed (diverse population); low entropy means concentration on few strategies. This connects to fleet diversity management.

## Quick Start

```rust
use ternary_fitness::{FitnessEvaluator, ExhaustiveSearch, FitnessLandscape, TernaryStrategy};

// Define a fitness function
let evaluator = FitnessEvaluator::new(|s: &TernaryStrategy| {
    s.values().iter().map(|&v| v as f64).sum() // simple: maximize sum
});

// Exhaustively evaluate all 3^5 = 243 strategies
let search = ExhaustiveSearch::new(5, evaluator);
let landscape = FitnessLandscape::from_search(&search);

println!("Peaks: {}", landscape.peaks().len());
println!("Best fitness: {}", landscape.best_fitness());
```

```bash
cargo add ternary-fitness
```

## API

| Type / Function | Description |
|---|---|
| `TernaryStrategy` | Vector of {-1, 0, +1} decisions |
| `FitnessEvaluator` | Wraps a fitness function `&Strategy → f64` |
| `ExhaustiveSearch` | Evaluates all 3ⁿ strategies |
| `FitnessLandscape` | Peaks, valleys, saddles, plateaus |
| `ParetoFront` | Non-dominated strategies |
| `Entropy` | Strategy distribution entropy |

## Architecture Notes

Fitness landscapes model the evolutionary dynamics of **SuperInstance** agent strategies. The γ + η = C conservation law constrains the landscape: strategies that maximize γ (growth) necessarily increase η (entropy cost), and the Pareto front traces the γ-η trade-off curve. Evolution on this landscape drives the fleet toward Pareto-optimal strategies. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Wright, Sewall. "The Roles of Mutation, Inbreeding, Crossbreeding, and Selection in Evolution," *Proc. 6th Int. Cong. Gen.*, 1932 — fitness landscapes.
- Kauffman, Stuart. *The Origins of Order*, Oxford UP, 1993 — NK landscapes and ruggedness.
- Deb, Kalyanmoy. *Multi-Objective Optimization Using Evolutionary Algorithms*, Wiley, 2001 — Pareto fronts.

## License

MIT
