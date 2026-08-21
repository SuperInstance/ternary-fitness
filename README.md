# Ternary Fitness — Landscape Analysis for Ternary Agent Strategy Evolution

**Ternary Fitness** analyzes the fitness landscape of ternary strategy spaces — the set of all strategies where each decision dimension takes a value in `{-1, 0, +1}`. It provides exhaustive landscape enumeration, topology analysis (peaks, saddles, basins of attraction), Pareto front computation for multi-objective optimization, and entropy / diversity measures.

## Why you would use this

When evolving a population of agents whose decisions are ternary (e.g. `{decelerate, coast, accelerate}`, `{sell, hold, buy}`, `{-1, 0, +1}` policy trits), the geometry of the fitness landscape determines which search algorithm will succeed:

- A **single-peak smooth landscape** is trivially solved by hill climbing.
- A **rugged landscape with many local optima** defeats naive climbers and demands evolutionary / annealing strategies.
- A **multi-objective landscape** has no single optimum — only a Pareto front of trade-offs.

This crate exhaustively characterizes such landscapes for small `n` (the strategy length): it enumerates all `3ⁿ` strategies, evaluates fitness against a per-state reward table, then reports peaks, saddle points, basins of attraction, the global optimum, and the multi-objective Pareto front. Use it to:

- pick the right optimizer for a given ternary control problem,
- verify that a learned / evolved strategy is actually Pareto-optimal,
- measure population diversity (entropy + pairwise Hamming distance) of a fleet of agents,
- instrument the **SuperInstance** ternary-agent ecosystem (this crate is the analysis half of strategy evolution; the runtime half lives elsewhere).

## How it works

### Fitness model

A strategy is a vector of `n` ternary values. An [`Environment`](crate::Environment) supplies a reward triple `[r(-1), r(0), r(+1)]` for each of the `n` states. The fitness of a strategy is the sum of the rewards for its chosen action at each state. This additive model makes the landscape separable, so the global optimum is trivially the per-state argmax — but the *topology* (peaks, saddles, basins, fronts) is still non-trivial and is what this crate characterizes.

### Exhaustive search

For small strategy spaces (`n ≤ 15` gives `3¹⁵ = 14,348,907` strategies), [`ExhaustiveSearch`](crate::ExhaustiveSearch) evaluates fitness for every ternary strategy in `O(3ⁿ)` time, giving the complete landscape with no sampling bias.

### Landscape topology

[`FitnessLandscape`](crate::FitnessLandscape) identifies:

- **Peaks**: strategies with no Hamming-1 neighbor of strictly higher fitness (local maxima).
- **Global peak**: the highest-fitness strategy.
- **Saddle points**: strategies that lie on the watershed between two or more distinct peaks (multiple peaks are reachable as uphill neighbors).
- **Basins of attraction**: the set of strategies that reach a given peak under steepest-ascent hill climbing.

The number of peaks is a ruggedness measure: more peaks = harder optimization.

### Pareto fronts

For multi-objective fitness, [`ParetoFront`](crate::ParetoFront) enumerates all strategies and returns the non-dominated set with respect to any subset of:

- `Objective::Reward` — cumulative reward (maximize),
- `Objective::Diversity` — Shannon entropy of the strategy's `{−1,0,+1}` distribution (maximize),
- `Objective::Speed` — number of zero choices (more zeros = fewer active decisions = "faster").

A solution is on the front if no other solution is at least as good on every objective and strictly better on at least one.

### Entropy

[`Entropy`](crate::Entropy) provides:

- `strategy_entropy`: Shannon entropy `H = −Σ p(v) · log₂ p(v)` over the three ternary values in a single strategy. Returns 0 for a constant strategy and `log₂ 3 ≈ 1.585` bits for a perfectly balanced one.
- `population_diversity`: average pairwise Hamming distance across a population of strategies.
- `normalized_entropy`: `strategy_entropy / log₂ 3`, in `[0, 1]`.

## Quick start

```bash
cargo add ternary-fitness
```

```rust
use ternary_fitness::{
    Entropy, Environment, ExhaustiveSearch, FitnessEvaluator, FitnessLandscape,
    Objective, ParetoFront, TernaryStrategy,
};

fn main() {
    // 1. Define an environment: per-state rewards for {-1, 0, +1}.
    //    state 0 rewards +1, state 1 rewards -1, state 2 rewards 0.
    let env = Environment::from_rows(&[
        [1.0, 0.5, 2.0], // state 0: +1 is best
        [3.0, 1.0, 0.0], // state 1: -1 is best
        [0.0, 4.0, 1.0], // state 2: 0 is best
    ]);

    // 2. Evaluate one strategy: cumulative reward across all states.
    let s = TernaryStrategy::new(vec![1, -1, 0]);
    println!("fitness of {s} = {}", FitnessEvaluator::evaluate(&s, &env)); // 9.0

    // 3. Find the global optimum by brute force over all 3^n strategies.
    let (best, best_fit) = ExhaustiveSearch::optimum(&env);
    println!("optimum = {best} fitness = {best_fit}"); // [+, -, 0] 9.0

    // 4. Build the full landscape: peaks, saddles, basins.
    let landscape = FitnessLandscape::build(&env);
    println!("landscape size = {}", landscape.size());               // 27
    println!("#peaks = {}", landscape.peaks().len());
    println!("global peak = {}", landscape.global_peak().unwrap().strategy);

    // 5. Multi-objective Pareto front: trade off reward vs diversity.
    let front = ParetoFront::compute(&env, &[Objective::Reward, Objective::Diversity]);
    println!("#pareto-optimal = {}", front.len());
    for sol in &front {
        println!("  {} reward={:.2} diversity={:.3}",
            sol.strategy, sol.reward, sol.diversity);
    }

    // 6. Entropy / diversity of a strategy and a population.
    println!("entropy of {s} = {:.3} bits", Entropy::strategy_entropy(&s));
    let pop = [
        TernaryStrategy::new(vec![ 1, -1,  0]),
        TernaryStrategy::new(vec![-1,  0,  1]),
        TernaryStrategy::new(vec![ 0,  0,  0]),
    ];
    println!("population diversity = {:.3}", Entropy::population_diversity(&pop));
}
```

Running the program above prints:

```
fitness of [+, -, 0] = 9
optimum = [+, -, 0] fitness = 9
landscape size = 27
#peaks = 1
global peak = [+, -, 0]
#pareto-optimal = 1
  [+, -, 0] reward=9.00 diversity=1.585
entropy of [+, -, 0] = 1.585 bits
population diversity = 2.333
```

## API

| Type / Function             | Purpose                                                              |
| --------------------------- | -------------------------------------------------------------------- |
| [`TernaryStrategy`](crate::TernaryStrategy)        | A vector of `{-1, 0, +1}` decisions with Hamming-1 neighbors.       |
| [`Environment`](crate::Environment)              | Per-state reward table `[r(-1), r(0), r(+1)]`.                       |
| [`FitnessEvaluator`](crate::FitnessEvaluator)      | Sums per-state rewards for a strategy under an environment.          |
| [`ExhaustiveSearch`](crate::ExhaustiveSearch)       | Enumerates and ranks all `3ⁿ` strategies.                            |
| [`FitnessLandscape`](crate::FitnessLandscape)       | Peaks, saddles, basins, global peak for a fixed environment.         |
| [`ParetoFront`](crate::ParetoFront)             | Non-dominated set across selected objectives.                        |
| [`Entropy`](crate::Entropy)                  | Strategy entropy, normalized entropy, Hamming distance, diversity.   |

## Architecture notes

Fitness landscapes model the evolutionary dynamics of **SuperInstance** agent strategies. The γ + η = C conservation law constrains the landscape: strategies that maximize γ (growth) necessarily increase η (entropy cost), and the Pareto front traces the γ–η trade-off curve. Evolution on this landscape drives the fleet toward Pareto-optimal strategies. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Wright, Sewall. "The Roles of Mutation, Inbreeding, Crossbreeding, and Selection in Evolution," *Proc. 6th Int. Cong. Gen.*, 1932 — fitness landscapes.
- Kauffman, Stuart. *The Origins of Order*, Oxford UP, 1993 — NK landscapes and ruggedness.
- Deb, Kalyanmoy. *Multi-Objective Optimization Using Evolutionary Algorithms*, Wiley, 2001 — Pareto fronts.

## License

MIT
