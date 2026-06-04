# ternary-fitness

Fitness landscape analysis for **ternary agent systems** — strategies composed of choices from {-1, 0, +1}.

## What is a Ternary Fitness Landscape?

A ternary fitness landscape maps every possible strategy (a sequence of -1, 0, or +1 choices) to a fitness value. Unlike binary landscapes (0/1), ternary landscapes introduce a **neutral middle ground** (0), creating richer topology:

- **Peaks**: Local maxima where no single-mutation neighbor has higher fitness
- **Basins**: Regions that drain toward the same local optimum
- **Saddle points**: Points between peaks where multiple uphill directions exist
- **Neutral networks**: Connected regions of equal fitness (from the 0 element)

## Quick Start

```rust
use ternary_fitness::*;

// Define an environment with 3 states, each with rewards for actions {-1, 0, +1}
let env = Environment::from_rows(&[
    [1.0, 0.5, 2.0],  // state 0: +1 is best
    [3.0, 1.0, 0.0],  // state 1: -1 is best
    [0.0, 4.0, 1.0],  // state 2: 0 is best
]);

// Evaluate a strategy
let strategy = TernaryStrategy::new(vec![1, -1, 0]);
let fitness = FitnessEvaluator::evaluate(&strategy, &env);
assert_eq!(fitness, 9.0); // 2.0 + 3.0 + 4.0

// Enumerate all 3^n strategies
let landscape = FitnessLandscape::build(&env);
println!("Global optimum fitness: {}", landscape.global_peak().unwrap().fitness);
println!("Number of peaks: {}", landscape.peaks().len());
```

## Core Concepts

### TernaryStrategy
A sequence of choices from {-1, 0, +1}. Strategies are mutated by changing one element at a time, giving each strategy `2n` neighbors (two choices per position).

### Environment
A reward matrix: `reward[state][action]` where action index maps to {-1, 0, +1}. Fitness is the cumulative reward across all states.

### FitnessLandscape
The complete landscape over all `3^n` strategies. Provides:
- **Peaks**: All local optima (no neighbor has higher fitness)
- **Basins**: Attraction regions for each peak
- **Saddle points**: Points adjacent to multiple peaks
- **Global optimum**: The highest-fitness strategy

### ParetoFront
Multi-objective optimization across:
- **Reward**: Cumulative environment reward
- **Diversity**: Shannon entropy of the strategy
- **Speed**: Number of non-zero actions (fewer = faster)

### Entropy
Shannon entropy measures strategy randomness:
- Maximum entropy (log₂ 3 ≈ 1.585) when all three values appear equally
- Zero entropy when all choices are identical

## Why Ternary?

Binary strategies are well-studied in evolutionary biology (Kauffman's NK landscapes). Ternary strategies add a **third option** — neither attack nor retreat, neither buy nor sell, neither cooperate nor defect. This middle ground creates:

1. **Neutral networks** — paths of equal fitness enabling drift
2. **Saddle points** — more common due to the flat middle
3 **Basin complexity** — larger basins with more internal structure

## License

MIT
