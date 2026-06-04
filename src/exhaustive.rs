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
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
        let env = Environment::from_rows(&[
            [1.0, 0.5, 2.0],
            [3.0, 1.0, 0.0],
        ]);
        let ranked = ExhaustiveSearch::ranked(&env);
        assert_eq!(ranked.len(), 9);
        // Verify descending order
        for window in ranked.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
    }

    #[test]
    fn test_optimum() {
        let env = Environment::from_rows(&[
            [1.0, 0.5, 2.0],
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ]);
        let (strategy, fitness) = ExhaustiveSearch::optimum(&env);
        assert_eq!(strategy.choices(), &[1, -1, 0]);
        assert_eq!(fitness, 9.0);
    }
}
