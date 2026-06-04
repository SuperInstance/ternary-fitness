/// An environment defining rewards for each (state, action) pair.
///
/// Actions map to ternary values: index 0 → -1, index 1 → 0, index 2 → +1.
#[derive(Clone, Debug)]
pub struct Environment {
    /// rewards[state][action_index] where action_index 0={-1}, 1={0}, 2={+1}
    rewards: Vec<[f64; 3]>,
}

impl Environment {
    /// Create an empty environment.
    pub fn new() -> Self {
        Self { rewards: Vec::new() }
    }

    /// Create from a slice of reward rows: `[[r(-1), r(0), r(+1)], ...]`.
    pub fn from_rows(rows: &[[f64; 3]]) -> Self {
        Self {
            rewards: rows.to_vec(),
        }
    }

    /// Number of states (strategy length this environment expects).
    pub fn num_states(&self) -> usize {
        self.rewards.len()
    }

    /// Get the reward for a given state and action.
    ///
    /// `action` must be in {-1, 0, +1}.
    pub fn reward(&self, state: usize, action: i8) -> f64 {
        let idx = (action + 1) as usize; // -1→0, 0→1, +1→2
        self.rewards[state][idx]
    }

    /// Get all rewards for a given state as [r(-1), r(0), r(+1)].
    pub fn state_rewards(&self, state: usize) -> [f64; 3] {
        self.rewards[state]
    }

    /// Add a state with rewards for {-1, 0, +1}.
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
        let env = Environment::from_rows(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);
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
