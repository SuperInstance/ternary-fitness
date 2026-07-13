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

/// A solution on the Pareto front: a strategy together with its scores on
/// each objective that the front was computed for.
#[derive(Clone, Debug)]
pub struct ParetoSolution {
    /// The strategy this solution represents.
    pub strategy: TernaryStrategy,
    /// Cumulative reward under the environment.
    pub reward: f64,
    /// Shannon entropy of the strategy's choice distribution, in bits.
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
    ///
    /// `a` dominates `b` iff `a` is at least as good as `b` on every objective
    /// and strictly better on at least one. The dominance relation is
    /// antisymmetric (a dominates b ⇒ b does not dominate a) and transitive
    /// (a dominates b and b dominates c ⇒ a dominates c); both properties
    /// follow from the componentwise `≥` plus a strict-inequality guard.
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
        Environment::from_rows(&[[1.0, 0.5, 2.0], [3.0, 1.0, 0.0]])
    }

    /// Helper: reconstruct every solution the way `compute` does internally,
    /// so tests can verify properties of the front against the full search
    /// space rather than against the front itself.
    fn all_solutions(env: &Environment) -> Vec<ParetoSolution> {
        let n = env.num_states();
        let total = 3usize.pow(n as u32);
        let mut out = Vec::with_capacity(total);
        for i in 0..total {
            let mut choices = Vec::with_capacity(n);
            let mut idx = i;
            for _ in 0..n {
                choices.push((idx % 3) as i8 - 1);
                idx /= 3;
            }
            let strategy = TernaryStrategy::new_unchecked(choices);
            out.push(ParetoSolution {
                reward: FitnessEvaluator::evaluate(&strategy, env),
                diversity: Entropy::strategy_entropy(&strategy),
                speed: n - strategy.active_count(),
                strategy,
            });
        }
        out
    }

    fn sol(reward: f64, diversity: f64, speed: usize) -> ParetoSolution {
        ParetoSolution {
            strategy: TernaryStrategy::new(vec![]),
            reward,
            diversity,
            speed,
        }
    }

    #[test]
    fn test_pareto_front_reward_only() {
        let front = ParetoFront::compute(&test_env(), &[Objective::Reward]);
        assert_eq!(front.len(), 1);
        // [1,-1] = env.reward(0,1) + env.reward(1,-1) = 2.0 + 3.0 = 5.0
        assert_eq!(front[0].reward, 5.0);
    }

    #[test]
    fn test_pareto_front_reward_and_diversity() {
        let front = ParetoFront::compute(&test_env(), &[Objective::Reward, Objective::Diversity]);
        // Should have multiple solutions trading off reward vs diversity
        assert!(!front.is_empty());
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
        assert!(!front.is_empty());
    }

    /// Real correctness check: every element on the front must be undominated
    /// by *any* solution in the entire search space (not just by other front
    /// members). The previous version of this test only compared pairs within
    /// the front, which is a vacuous tautology because dominance is
    /// antisymmetric by construction — it passed even when `compute` was
    /// sabotaged to return every solution.
    #[test]
    fn test_front_members_undominated_by_any_solution() {
        let objectives = [Objective::Reward, Objective::Diversity];
        let front = ParetoFront::compute(&test_env(), &objectives);
        let universe = all_solutions(&test_env());

        for f in &front {
            let dominated = universe.iter().any(|other| {
                other.strategy != f.strategy && ParetoFront::dominates(other, f, &objectives)
            });
            assert!(
                !dominated,
                "front member {:?} is dominated by some solution in the search space",
                f.strategy
            );
        }
    }

    /// Real correctness check the other direction: every solution NOT on the
    /// front must be dominated by at least one front member. Together with
    /// the previous test this fully characterizes the Pareto front.
    #[test]
    fn test_non_front_solutions_are_dominated_by_front() {
        let objectives = [Objective::Reward, Objective::Diversity];
        let front = ParetoFront::compute(&test_env(), &objectives);
        let universe = all_solutions(&test_env());

        for s in &universe {
            if front.iter().any(|f| f.strategy == s.strategy) {
                continue;
            }
            let dominated = front
                .iter()
                .any(|f| ParetoFront::dominates(f, s, &objectives));
            assert!(
                dominated,
                "non-front solution {:?} is not dominated by any front member",
                s.strategy
            );
        }
    }

    /// Worked-example check of antisymmetry on a hand-built triple.
    /// A=(5,1.0) dominates B=(5,0.5) (strictly better in diversity, equal in reward),
    /// but B must not dominate A.
    #[test]
    fn test_dominance_antisymmetric_on_worked_example() {
        let objs = [Objective::Reward, Objective::Diversity];
        let a = sol(5.0, 1.0, 0);
        let b = sol(5.0, 0.5, 0);
        assert!(ParetoFront::dominates(&a, &b, &objs));
        assert!(!ParetoFront::dominates(&b, &a, &objs));

        // Incomparable pair: neither dominates.
        let c = sol(4.0, 1.5, 0);
        assert!(!ParetoFront::dominates(&a, &c, &objs));
        assert!(!ParetoFront::dominates(&c, &a, &objs));
    }

    /// Identical solutions must NOT dominate each other — dominance requires
    /// strict improvement in at least one objective. This guards against the
    /// classic `>=` for `>` regression, which is invisible on inputs that
    /// happen to have no equal pairs.
    #[test]
    fn test_dominance_equal_solutions_do_not_dominate() {
        let objs = [Objective::Reward, Objective::Diversity];
        let a = sol(5.0, 1.0, 0);
        let b = sol(5.0, 1.0, 0);
        assert!(!ParetoFront::dominates(&a, &b, &objs));
        assert!(!ParetoFront::dominates(&b, &a, &objs));
    }

    /// Worked-example check of transitivity: if A dominates B and B dominates C
    /// then A must dominate C.
    #[test]
    fn test_dominance_transitive_on_worked_example() {
        let objs = [Objective::Reward, Objective::Diversity];
        let a = sol(5.0, 2.0, 0);
        let b = sol(4.0, 1.5, 0);
        let c = sol(3.0, 1.0, 0);
        assert!(ParetoFront::dominates(&a, &b, &objs));
        assert!(ParetoFront::dominates(&b, &c, &objs));
        assert!(ParetoFront::dominates(&a, &c, &objs));
    }

    /// Antisymmetry + transitivity checked exhaustively over a real computed
    /// front. Antisymmetry is structurally guaranteed; this guards against a
    /// future regression that introduces e.g. a `<=` where `<` was meant.
    #[test]
    fn test_dominance_properties_on_real_front() {
        let objs = [Objective::Reward, Objective::Diversity, Objective::Speed];
        let universe = all_solutions(&test_env());

        // Antisymmetry: no pair (i, j) with i != j mutually dominates.
        for i in 0..universe.len() {
            for j in (i + 1)..universe.len() {
                let fwd = ParetoFront::dominates(&universe[i], &universe[j], &objs);
                let bwd = ParetoFront::dominates(&universe[j], &universe[i], &objs);
                assert!(!(fwd && bwd), "mutual domination between {i} and {j}");
            }
        }

        // Transitivity sample: pick every triple where i<j<k and verify.
        // (3^2 = 9 solutions ⇒ 84 triples, cheap.)
        for i in 0..universe.len() {
            for j in (i + 1)..universe.len() {
                if !ParetoFront::dominates(&universe[i], &universe[j], &objs) {
                    continue;
                }
                for k in (j + 1)..universe.len() {
                    if ParetoFront::dominates(&universe[j], &universe[k], &objs) {
                        assert!(
                            ParetoFront::dominates(&universe[i], &universe[k], &objs),
                            "transitivity violation: {i} dom {j} dom {k} but not {i} dom {k}"
                        );
                    }
                }
            }
        }
    }

    /// Edge case: empty environment. 3^0 = 1 strategy (the empty strategy),
    /// so the front is exactly that single strategy — undominated.
    #[test]
    fn test_pareto_front_empty_environment() {
        let env = Environment::new();
        let front = ParetoFront::compute(&env, &[Objective::Reward]);
        assert_eq!(front.len(), 1);
        assert!(front[0].strategy.is_empty());
    }
}
