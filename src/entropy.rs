//! Shannon entropy for ternary strategies and population diversity.

/// Shannon entropy computation for ternary strategies.
pub struct Entropy;

impl Entropy {
    /// Compute the Shannon entropy of a single strategy.
    ///
    /// Measures the information content based on the distribution of {-1, 0, +1}
    /// values. Returns entropy in bits.
    ///
    /// - 0.0 if all values are identical
    /// - log₂(3) ≈ 1.585 if all three values appear equally
    pub fn strategy_entropy(strategy: &crate::TernaryStrategy) -> f64 {
        let choices = strategy.choices();
        let n = choices.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let count = |val: i8| choices.iter().filter(|&&c| c == val).count() as f64;

        let p_neg = count(-1) / n;
        let p_zero = count(0) / n;
        let p_pos = count(1) / n;

        let entropy = |p: f64| if p > 0.0 { -p * p.log2() } else { 0.0 };

        entropy(p_neg) + entropy(p_zero) + entropy(p_pos)
    }

    /// Maximum possible entropy for a ternary strategy: log₂(3) ≈ 1.585 bits.
    pub fn max_entropy() -> f64 {
        3.0_f64.log2()
    }

    /// Compute population diversity as the average pairwise entropy distance.
    ///
    /// Measures how diverse a set of strategies is.
    pub fn population_diversity(strategies: &[crate::TernaryStrategy]) -> f64 {
        if strategies.len() <= 1 {
            return 0.0;
        }

        let _n = strategies.len() as f64;
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..strategies.len() {
            for j in (i + 1)..strategies.len() {
                total_distance += Self::hamming_distance(&strategies[i], &strategies[j]);
                count += 1;
            }
        }

        total_distance / count as f64
    }

    /// Hamming distance between two strategies (number of differing positions).
    pub fn hamming_distance(a: &crate::TernaryStrategy, b: &crate::TernaryStrategy) -> f64 {
        a.choices()
            .iter()
            .zip(b.choices().iter())
            .filter(|(x, y)| x != y)
            .count() as f64
    }

    /// Normalized entropy: strategy entropy / max entropy, in [0, 1].
    pub fn normalized_entropy(strategy: &crate::TernaryStrategy) -> f64 {
        let max = Self::max_entropy();
        if max == 0.0 {
            return 0.0;
        }
        Self::strategy_entropy(strategy) / max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TernaryStrategy;

    #[test]
    fn test_zero_entropy_uniform() {
        let s = TernaryStrategy::new(vec![1, 1, 1]);
        assert_eq!(Entropy::strategy_entropy(&s), 0.0);
    }

    #[test]
    fn test_zero_entropy_all_neg() {
        let s = TernaryStrategy::new(vec![-1, -1]);
        assert_eq!(Entropy::strategy_entropy(&s), 0.0);
    }

    #[test]
    fn test_max_entropy_balanced() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        let entropy = Entropy::strategy_entropy(&s);
        let expected = 3.0_f64.log2();
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_partial_entropy() {
        let s = TernaryStrategy::new(vec![-1, 0, 0, 0]);
        let entropy = Entropy::strategy_entropy(&s);
        // p(-1)=0.25, p(0)=0.75, p(1)=0
        let expected = -0.25 * (0.25_f64).log2() + -0.75 * (0.75_f64).log2();
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_max_entropy_constant() {
        assert!((Entropy::max_entropy() - 1.585).abs() < 0.001);
    }

    #[test]
    fn test_hamming_distance_same() {
        let a = TernaryStrategy::new(vec![-1, 0, 1]);
        assert_eq!(Entropy::hamming_distance(&a, &a), 0.0);
    }

    #[test]
    fn test_hamming_distance_different() {
        let a = TernaryStrategy::new(vec![-1, 0, 1]);
        let b = TernaryStrategy::new(vec![1, 0, -1]);
        assert_eq!(Entropy::hamming_distance(&a, &b), 2.0);
    }

    #[test]
    fn test_population_diversity_empty() {
        assert_eq!(Entropy::population_diversity(&[]), 0.0);
    }

    #[test]
    fn test_population_diversity_single() {
        let s = TernaryStrategy::new(vec![0]);
        assert_eq!(Entropy::population_diversity(&[s]), 0.0);
    }

    #[test]
    fn test_normalized_entropy() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        let norm = Entropy::normalized_entropy(&s);
        assert!((norm - 1.0).abs() < 1e-10);
    }
}
