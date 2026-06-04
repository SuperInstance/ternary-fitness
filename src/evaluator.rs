//! Fitness evaluation for ternary strategies.

use crate::{Environment, TernaryStrategy};

/// Evaluates the fitness of a ternary strategy against an environment.
///
/// Fitness is the cumulative reward: sum of `reward(state_i, action_i)` for each
/// position `i` in the strategy.
pub struct FitnessEvaluator;

impl FitnessEvaluator {
    /// Evaluate a strategy's fitness against an environment.
    ///
    /// The strategy length must match the number of states in the environment.
    pub fn evaluate(strategy: &TernaryStrategy, env: &Environment) -> f64 {
        assert_eq!(
            strategy.len(),
            env.num_states(),
            "Strategy length ({}) must match environment states ({})",
            strategy.len(),
            env.num_states()
        );

        strategy
            .choices()
            .iter()
            .enumerate()
            .map(|(state, &action)| env.reward(state, action))
            .sum()
    }

    /// Evaluate fitness for multiple strategies.
    pub fn evaluate_all<'a>(
        strategies: impl IntoIterator<Item = &'a TernaryStrategy>,
        env: &Environment,
    ) -> Vec<f64> {
        strategies
            .into_iter()
            .map(|s| Self::evaluate(s, env))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_env() -> Environment {
        Environment::from_rows(&[
            [1.0, 0.5, 2.0], // state 0: +1 is best
            [3.0, 1.0, 0.0], // state 1: -1 is best
            [0.0, 4.0, 1.0], // state 2: 0 is best
        ])
    }

    #[test]
    fn test_optimal_strategy() {
        let env = simple_env();
        let strategy = TernaryStrategy::new(vec![1, -1, 0]);
        assert_eq!(FitnessEvaluator::evaluate(&strategy, &env), 9.0);
    }

    #[test]
    fn test_worst_strategy() {
        let env = simple_env();
        let strategy = TernaryStrategy::new(vec![-1, 1, 1]);
        assert_eq!(FitnessEvaluator::evaluate(&strategy, &env), 2.0);
    }

    #[test]
    fn test_all_zero_strategy() {
        let env = simple_env();
        let strategy = TernaryStrategy::new(vec![0, 0, 0]);
        assert_eq!(FitnessEvaluator::evaluate(&strategy, &env), 5.5);
    }

    #[test]
    fn test_evaluate_all() {
        let env = simple_env();
        let strategies = vec![
            TernaryStrategy::new(vec![1, -1, 0]),
            TernaryStrategy::new(vec![-1, 1, 1]),
        ];
        let fitnesses = FitnessEvaluator::evaluate_all(&strategies, &env);
        assert_eq!(fitnesses, vec![9.0, 2.0]);
    }
}
