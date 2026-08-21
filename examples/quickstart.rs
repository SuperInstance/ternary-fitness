//! Quick-start example for `ternary-fitness`.
//!
//! Run with `cargo run --example quickstart`.

use ternary_fitness::{
    Entropy, Environment, ExhaustiveSearch, FitnessEvaluator, FitnessLandscape, Objective,
    ParetoFront, TernaryStrategy,
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
    println!("landscape size = {}", landscape.size()); // 27
    println!("#peaks = {}", landscape.peaks().len());
    println!(
        "global peak = {}",
        landscape.global_peak().unwrap().strategy
    );

    // 5. Multi-objective Pareto front: trade off reward vs diversity.
    let front = ParetoFront::compute(&env, &[Objective::Reward, Objective::Diversity]);
    println!("#pareto-optimal = {}", front.len());
    for sol in &front {
        println!(
            "  {} reward={:.2} diversity={:.3}",
            sol.strategy, sol.reward, sol.diversity
        );
    }

    // 6. Entropy / diversity of a strategy and a population.
    println!("entropy of {s} = {:.3} bits", Entropy::strategy_entropy(&s));
    let pop = [
        TernaryStrategy::new(vec![1, -1, 0]),
        TernaryStrategy::new(vec![-1, 0, 1]),
        TernaryStrategy::new(vec![0, 0, 0]),
    ];
    println!(
        "population diversity = {:.3}",
        Entropy::population_diversity(&pop)
    );
}
