/// A strategy composed of ternary choices: {-1, 0, +1}.
///
/// Each element in the strategy corresponds to an action taken in a sequential
/// decision process. The strategy's fitness is determined by an [`Environment`](crate::Environment).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TernaryStrategy {
    /// The choices, each in {-1, 0, +1}.
    choices: Vec<i8>,
}

impl TernaryStrategy {
    /// Create a new strategy from a vector of {-1, 0, +1} values.
    ///
    /// # Panics
    /// Panics if any value is not in {-1, 0, +1}.
    pub fn new(choices: Vec<i8>) -> Self {
        for &c in &choices {
            assert!(
                c == -1 || c == 0 || c == 1,
                "Invalid ternary value: {c}. Must be -1, 0, or +1."
            );
        }
        Self { choices }
    }

    /// Create a strategy without validation. Caller must ensure values are in {-1, 0, +1}.
    pub(crate) fn new_unchecked(choices: Vec<i8>) -> Self {
        Self { choices }
    }

    /// The number of decisions in this strategy.
    pub fn len(&self) -> usize {
        self.choices.len()
    }

    /// Whether this is an empty strategy.
    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }

    /// Access the choices as a slice.
    pub fn choices(&self) -> &[i8] {
        &self.choices
    }

    /// Get the choice at a given position.
    pub fn get(&self, index: usize) -> Option<i8> {
        self.choices.get(index).copied()
    }

    /// Generate all neighbors by mutating exactly one position.
    ///
    /// Each position can change to 2 other values, so a strategy of length `n`
    /// has `2n` neighbors.
    pub fn neighbors(&self) -> Vec<TernaryStrategy> {
        let mut result = Vec::with_capacity(self.choices.len() * 2);
        let alternatives: &[i8] = &[-1, 0, 1];

        for i in 0..self.choices.len() {
            for &alt in alternatives {
                if alt != self.choices[i] {
                    let mut neighbor = self.choices.clone();
                    neighbor[i] = alt;
                    result.push(TernaryStrategy::new_unchecked(neighbor));
                }
            }
        }
        result
    }

    /// Mutate a single position to a specific value.
    pub fn with_mutation(&self, position: usize, new_value: i8) -> Option<TernaryStrategy> {
        if position >= self.choices.len() || new_value == self.choices[position] {
            return None;
        }
        assert!(new_value == -1 || new_value == 0 || new_value == 1);
        let mut mutated = self.choices.clone();
        mutated[position] = new_value;
        Some(TernaryStrategy::new_unchecked(mutated))
    }

    /// Count the number of non-zero choices.
    pub fn active_count(&self) -> usize {
        self.choices.iter().filter(|&&c| c != 0).count()
    }
}

impl std::fmt::Display for TernaryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display: Vec<&str> = self
            .choices
            .iter()
            .map(|&c| match c {
                -1 => "-",
                0 => "0",
                1 => "+",
                _ => "?",
            })
            .collect();
        write!(f, "[{}]", display.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        assert_eq!(s.choices(), &[-1, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "Invalid ternary value")]
    fn test_new_invalid() {
        TernaryStrategy::new(vec![2]);
    }

    #[test]
    fn test_len_and_empty() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());

        let empty = TernaryStrategy::new(vec![]);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_neighbors_count() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        let neighbors = s.neighbors();
        // 3 positions × 2 alternatives = 6 neighbors
        assert_eq!(neighbors.len(), 6);
    }

    #[test]
    fn test_neighbors_all_different() {
        let s = TernaryStrategy::new(vec![0]);
        let neighbors = s.neighbors();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&TernaryStrategy::new(vec![-1])));
        assert!(neighbors.contains(&TernaryStrategy::new(vec![1])));
    }

    #[test]
    fn test_display() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        assert_eq!(format!("{s}"), "[-, 0, +]");
    }

    #[test]
    fn test_active_count() {
        let s = TernaryStrategy::new(vec![-1, 0, 1, 0]);
        assert_eq!(s.active_count(), 2);
    }

    #[test]
    fn test_with_mutation() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        let mutated = s.with_mutation(1, 1).unwrap();
        assert_eq!(mutated.choices(), &[-1, 1, 1]);
    }

    #[test]
    fn test_with_mutation_no_change() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        assert!(s.with_mutation(0, -1).is_none());
    }
}
