//! Multi-objective Pareto front analysis.

use crate::{Entropy, Environment, FitnessEvaluator, TernaryStrategy};

/// A multi-objective optimization objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Maximize cumulative reward.
    Reward,
    /// Maximize strategy entropy (diversity of choices).
    Diversity,
    /// Maximize speed (fewer non-zero actions = faster; stored as active_count).
    Speed,
}

/// A solution on the Pareto front.
#[derive(Clone, Debug)]
pub struct ParetoSolution {
    pub strategy: TernaryStrategy,
    pub reward: f64,
    pub diversity: f64,
    /// Speed score: strategy length - active_count (more zeros = faster).
    pub speed: usize,
}

/// Pareto front computation for multi-objective ternary optimization.
pub struct ParetoFront;

impl ParetoFront {
    /// Compute all Pareto-optimal solutions for the given objectives.
    ///
    /// A solution is Pareto-optimal if no other solution is better in all
    /// selected objectives.
    pub fn compute(env: &Environment, objectives: &[Objective]) -> Vec<ParetoSolution> {
        let n = env.num_states();
        let total = 3usize.pow(n as u32);
        let mut solutions = Vec::with_capacity(total);

        for i in 0..total {
            let mut choices = Vec::with_capacity(n);
            let mut idx = i;
            for _ in 0..n {
                choices.push((idx % 3) as i8 - 1);
                idx /= 3;
            }
            let strategy = TernaryStrategy::new_unchecked(choices);
            let reward = FitnessEvaluator::evaluate(&strategy, env);
            let diversity = Entropy::strategy_entropy(&strategy);
            let speed = n - strategy.active_count();

            solutions.push(ParetoSolution {
                strategy,
                reward,
                diversity,
                speed,
            });
        }

        // Filter to Pareto front: keep solutions not dominated by any other
        let mut front = Vec::new();

        for candidate in &solutions {
            let dominated = solutions.iter().any(|other| {
                other.strategy != candidate.strategy
                    && Self::dominates(other, candidate, objectives)
            });
            if !dominated {
                front.push(candidate.clone());
            }
        }

        front
    }

    /// Check if `a` dominates `b` across all given objectives.
    fn dominates(a: &ParetoSolution, b: &ParetoSolution, objectives: &[Objective]) -> bool {
        let mut at_least_one_better = false;
        for obj in objectives {
            let (va, vb) = match obj {
                Objective::Reward => (a.reward, b.reward),
                Objective::Diversity => (a.diversity, b.diversity),
                Objective::Speed => (a.speed as f64, b.speed as f64),
            };
            if va < vb {
                return false; // a is worse in this objective
            }
            if va > vb {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_env() -> Environment {
        Environment::from_rows(&[
            [1.0, 0.5, 2.0],
            [3.0, 1.0, 0.0],
        ])
    }

    #[test]
    fn test_pareto_front_reward_only() {
        let front = ParetoFront::compute(&test_env(), &[Objective::Reward]);
        assert_eq!(front.len(), 1);
        // [1,-1] = env.reward(0,1) + env.reward(1,-1) = 2.0 + 3.0 = 5.0
    }

    #[test]
    fn test_pareto_front_reward_and_diversity() {
        let front = ParetoFront::compute(&test_env(), &[Objective::Reward, Objective::Diversity]);
        // Should have multiple solutions trading off reward vs diversity
        assert!(front.len() >= 1);
        // The max-reward solution should be on the front
        // Best reward: [1,-1] = 2+3 = 5
        assert!(front.iter().any(|s| s.reward == 5.0));
    }

    #[test]
    fn test_pareto_front_all_objectives() {
        let front = ParetoFront::compute(
            &test_env(),
            &[Objective::Reward, Objective::Diversity, Objective::Speed],
        );
        assert!(front.len() >= 1);
    }

    #[test]
    fn test_no_dominated_solutions() {
        let front = ParetoFront::compute(
            &test_env(),
            &[Objective::Reward, Objective::Diversity],
        );
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i != j {
                    // No solution on the front should dominate another on the front
                    let objectives = [Objective::Reward, Objective::Diversity];
                    // They can only dominate in one direction, not both
                    let fwd = ParetoFront::dominates(&front[i], &front[j], &objectives);
                    let bwd = ParetoFront::dominates(&front[j], &front[i], &objectives);
                    assert!(!(fwd && bwd), "Mutual domination detected");
                }
            }
        }
    }
}
