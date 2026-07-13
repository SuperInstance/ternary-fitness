//! Exhaustive enumeration of all ternary strategies.

use crate::{Environment, FitnessEvaluator, TernaryStrategy};

/// Exhaustively enumerates all 3^n possible ternary strategies and evaluates them.
pub struct ExhaustiveSearch;

impl ExhaustiveSearch {
    /// Generate all 3^n ternary strategies of a given length.
    ///
    /// For n=4, this produces 81 strategies.
    pub fn enumerate(n: usize) -> Vec<TernaryStrategy> {
        let total = 3usize.pow(n as u32);
        let mut strategies = Vec::with_capacity(total);

        for i in 0..total {
            let mut choices = Vec::with_capacity(n);
            let mut idx = i;
            for _ in 0..n {
                choices.push((idx % 3) as i8 - 1); // 0→-1, 1→0, 2→+1
                idx /= 3;
            }
            strategies.push(TernaryStrategy::new_unchecked(choices));
        }

        strategies
    }

    /// Evaluate all strategies and return them sorted by fitness (descending).
    pub fn ranked(env: &Environment) -> Vec<(TernaryStrategy, f64)> {
        let n = env.num_states();
        let strategies = Self::enumerate(n);
        let mut ranked: Vec<(TernaryStrategy, f64)> = strategies
            .into_iter()
            .map(|s| {
                let fitness = FitnessEvaluator::evaluate(&s, env);
                (s, fitness)
            })
            .collect();
        // Use total_cmp so NaN sorts deterministically (as the smallest value)
        // instead of panicking or silently treating all NaNs as equal.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
    }

    /// Find the global optimum (best strategy).
    pub fn optimum(env: &Environment) -> (TernaryStrategy, f64) {
        Self::ranked(env)
            .into_iter()
            .next()
            .expect("Environment has no states")
    }

    /// Count the total number of strategies for a given length.
    pub fn count(n: usize) -> usize {
        3usize.pow(n as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_n1() {
        let strategies = ExhaustiveSearch::enumerate(1);
        assert_eq!(strategies.len(), 3);
        assert!(strategies.contains(&TernaryStrategy::new(vec![-1])));
        assert!(strategies.contains(&TernaryStrategy::new(vec![0])));
        assert!(strategies.contains(&TernaryStrategy::new(vec![1])));
    }

    #[test]
    fn test_enumerate_n2() {
        let strategies = ExhaustiveSearch::enumerate(2);
        assert_eq!(strategies.len(), 9);
    }

    #[test]
    fn test_enumerate_n3() {
        let strategies = ExhaustiveSearch::enumerate(3);
        assert_eq!(strategies.len(), 27);
        // Check uniqueness
        let set: std::collections::HashSet<_> = strategies.iter().collect();
        assert_eq!(set.len(), 27);
    }

    #[test]
    fn test_enumerate_n4_81_strategies() {
        let strategies = ExhaustiveSearch::enumerate(4);
        assert_eq!(strategies.len(), 81);
        // Verify uniqueness
        let set: std::collections::HashSet<_> = strategies.iter().collect();
        assert_eq!(set.len(), 81);
        // All should be valid length-4 strategies
        for s in &strategies {
            assert_eq!(s.len(), 4);
        }
    }

    #[test]
    fn test_count() {
        assert_eq!(ExhaustiveSearch::count(4), 81);
        assert_eq!(ExhaustiveSearch::count(1), 3);
        assert_eq!(ExhaustiveSearch::count(0), 1);
    }

    #[test]
    fn test_ranked_ordering() {
        let env = Environment::from_rows(&[[1.0, 0.5, 2.0], [3.0, 1.0, 0.0]]);
        let ranked = ExhaustiveSearch::ranked(&env);
        assert_eq!(ranked.len(), 9);
        // Verify descending order
        for window in ranked.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
    }

    #[test]
    fn test_optimum() {
        let env = Environment::from_rows(&[[1.0, 0.5, 2.0], [3.0, 1.0, 0.0], [0.0, 4.0, 1.0]]);
        let (strategy, fitness) = ExhaustiveSearch::optimum(&env);
        assert_eq!(strategy.choices(), &[1, -1, 0]);
        assert_eq!(fitness, 9.0);
    }

    /// Stronger completeness check: for n=5, enumerate must produce all 3^5
    /// = 243 distinct ternary vectors and nothing else. This is the
    /// invariant that justifies calling the search "exhaustive".
    #[test]
    fn test_enumerate_n5_covers_full_space() {
        let strategies = ExhaustiveSearch::enumerate(5);
        assert_eq!(strategies.len(), 243);

        // Collect every observed vector.
        let observed: std::collections::HashSet<Vec<i8>> =
            strategies.iter().map(|s| s.choices().to_vec()).collect();
        assert_eq!(observed.len(), 243, "duplicate vectors in enumeration");

        // Independently generate the full Cartesian product {-1,0,1}^5 and
        // confirm every element appears.
        for &a in &[-1, 0, 1] {
            for &b in &[-1, 0, 1] {
                for &c in &[-1, 0, 1] {
                    for &d in &[-1, 0, 1] {
                        for &e in &[-1, 0, 1] {
                            assert!(
                                observed.contains(&vec![a, b, c, d, e]),
                                "missing vector {a:?}{b:?}{c:?}{d:?}{e:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Edge case: n=0. 3^0 = 1 strategy — the empty strategy. The search
    /// space is a singleton containing only the empty vector.
    #[test]
    fn test_enumerate_n0_single_empty_strategy() {
        let strategies = ExhaustiveSearch::enumerate(0);
        assert_eq!(strategies.len(), 1);
        assert!(strategies[0].is_empty());
    }

    /// Edge case: empty environment. `ranked` and `optimum` must still
    /// return a one-element list (the empty strategy with fitness 0).
    #[test]
    fn test_ranked_empty_environment() {
        let env = Environment::new();
        let ranked = ExhaustiveSearch::ranked(&env);
        assert_eq!(ranked.len(), 1);
        assert!(ranked[0].0.is_empty());
        assert_eq!(ranked[0].1, 0.0);

        let (best, fit) = ExhaustiveSearch::optimum(&env);
        assert!(best.is_empty());
        assert_eq!(fit, 0.0);
    }

    /// NaN safety: if any reward is NaN, `ranked` must not panic. Pre-fix the
    /// code used `.partial_cmp().unwrap()` which would crash on NaN inputs;
    /// now it uses `total_cmp` and produces a total order. (Under
    /// `f64::total_cmp`, NaN sorts as the *largest* value, so a
    /// descending-order `ranked` will list NaN-contaminated entries first —
    /// this is intentional and deterministic, not a panic.)
    #[test]
    fn test_ranked_with_nan_does_not_panic() {
        let env = Environment::from_rows(&[[1.0, f64::NAN, 2.0], [3.0, 1.0, 0.0]]);
        let ranked = ExhaustiveSearch::ranked(&env);
        assert_eq!(ranked.len(), 9);
        // Verify the result is a total order: every adjacent pair satisfies
        // `ranked[i] >= ranked[i+1]` under `total_cmp`. Pre-fix this would
        // have panicked inside `partial_cmp().unwrap()`.
        for w in ranked.windows(2) {
            assert!(w[0].1.total_cmp(&w[1].1) != std::cmp::Ordering::Less);
        }
        // Sanity: the best *non-NaN* fitness is still 5.0 (state0=+1, state1=-1).
        let best_finite = ranked
            .iter()
            .map(|(_, f)| *f)
            .filter(|f| f.is_finite())
            .max_by(f64::total_cmp)
            .unwrap();
        assert_eq!(best_finite, 5.0);
    }
}
