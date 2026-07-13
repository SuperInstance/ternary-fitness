//! Environment definition for fitness evaluation.

/// An environment defining rewards for each (state, action) pair.
///
/// A strategy of length `n` is evaluated against an environment with `n`
/// states. Each state has three rewards, indexed by ternary action value:
/// index 0 → action `-1`, index 1 → action `0`, index 2 → action `+1`.
///
/// Environments are constructed either empty ([`Environment::new`]) and grown
/// one state at a time with [`Environment::add_state`], or all at once from a
/// slice of `[r(-1), r(0), r(+1)]` triples with [`Environment::from_rows`].
#[derive(Clone, Debug)]
pub struct Environment {
    /// `rewards[state][action_index]` where `action_index` is `0={-1}`, `1={0}`, `2={+1}`.
    rewards: Vec<[f64; 3]>,
}

impl Environment {
    /// Create an empty environment (zero states).
    pub fn new() -> Self {
        Self {
            rewards: Vec::new(),
        }
    }

    /// Create from a slice of reward rows, where each row is `[r(-1), r(0), r(+1)]`.
    ///
    /// Row `i` becomes the rewards for state `i`. The number of states is
    /// `rows.len()`.
    pub fn from_rows(rows: &[[f64; 3]]) -> Self {
        Self {
            rewards: rows.to_vec(),
        }
    }

    /// Number of states — equivalently, the strategy length this environment expects.
    pub fn num_states(&self) -> usize {
        self.rewards.len()
    }

    /// Get the reward for taking `action` in `state`.
    ///
    /// `action` must be in `{-1, 0, +1}` and `state` must be `< num_states()`.
    /// Other values will trigger an out-of-bounds panic — this is treated as
    /// a programming error since [`TernaryStrategy`] only ever holds valid
    /// ternary values.
    pub fn reward(&self, state: usize, action: i8) -> f64 {
        let idx = (action + 1) as usize; // -1→0, 0→1, +1→2
        self.rewards[state][idx]
    }

    /// Get all rewards for a given state as `[r(-1), r(0), r(+1)]`.
    ///
    /// `state` must be `< num_states()`.
    pub fn state_rewards(&self, state: usize) -> [f64; 3] {
        self.rewards[state]
    }

    /// Append a new state with the given `[r(-1), r(0), r(+1)]` reward triple.
    pub fn add_state(&mut self, rewards: [f64; 3]) {
        self.rewards.push(rewards);
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_rows() {
        let env = Environment::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(env.num_states(), 2);
        assert_eq!(env.reward(0, -1), 1.0);
        assert_eq!(env.reward(0, 0), 2.0);
        assert_eq!(env.reward(0, 1), 3.0);
        assert_eq!(env.reward(1, -1), 4.0);
        assert_eq!(env.reward(1, 1), 6.0);
    }

    #[test]
    fn test_add_state() {
        let mut env = Environment::new();
        env.add_state([1.0, 2.0, 3.0]);
        assert_eq!(env.num_states(), 1);
        assert_eq!(env.reward(0, 0), 2.0);
    }

    #[test]
    fn test_default() {
        let env = Environment::default();
        assert_eq!(env.num_states(), 0);
    }
}
