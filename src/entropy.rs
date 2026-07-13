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

    /// Compute population diversity as the average pairwise Hamming distance.
    ///
    /// Returns 0.0 for populations of size 0 or 1. For larger populations,
    /// averages the Hamming distance over every unordered pair of distinct
    /// strategies — so two identical strategies contribute 0 to the average
    /// and an all-identical population reports 0 diversity.
    pub fn population_diversity(strategies: &[crate::TernaryStrategy]) -> f64 {
        if strategies.len() <= 1 {
            return 0.0;
        }

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

    /// Edge case: empty strategy. Entropy is 0 (no information), and
    /// `normalized_entropy` is 0/0 in spirit but the impl guards the divide.
    #[test]
    fn test_entropy_empty_strategy() {
        let s = TernaryStrategy::new(vec![]);
        assert_eq!(Entropy::strategy_entropy(&s), 0.0);
        assert_eq!(Entropy::normalized_entropy(&s), 0.0);
    }

    /// Edge case: single-choice strategy. One symbol ⇒ entropy 0.
    #[test]
    fn test_entropy_single_choice() {
        let s = TernaryStrategy::new(vec![0]);
        assert_eq!(Entropy::strategy_entropy(&s), 0.0);
    }

    /// Edge case: all-identical population reports 0 diversity.
    #[test]
    fn test_population_diversity_all_identical() {
        let s = TernaryStrategy::new(vec![-1, 0, 1]);
        assert_eq!(
            Entropy::population_diversity(&[s.clone(), s.clone(), s.clone()]),
            0.0
        );
    }

    /// Edge case: unequal-length strategies. Hamming distance zips only to
    /// the shorter length — verify the documented behavior so it doesn't
    /// silently change.
    #[test]
    fn test_hamming_distance_unequal_length() {
        let a = TernaryStrategy::new(vec![-1, 0, 1]);
        let b = TernaryStrategy::new(vec![-1, 0]);
        // zip stops at min(3, 2) = 2; both positions agree ⇒ distance 0.
        assert_eq!(Entropy::hamming_distance(&a, &b), 0.0);

        let c = TernaryStrategy::new(vec![1, 0]);
        // pos 0 differs (-1 vs 1), pos 1 agrees ⇒ distance 1.
        assert_eq!(Entropy::hamming_distance(&a, &c), 1.0);
    }

    /// Worked-example check of `population_diversity`:
    /// strategies [1,-1,0], [-1,0,1], [0,0,0].
    /// Pairs:
    ///   [1,-1,0] vs [-1,0,1]: 3 differences
    ///   [1,-1,0] vs [0,0,0]:  2 differences
    ///   [-1,0,1] vs [0,0,0]:  2 differences
    /// Mean = (3 + 2 + 2) / 3 = 7/3 ≈ 2.3333
    #[test]
    fn test_population_diversity_worked_example() {
        let pop = [
            TernaryStrategy::new(vec![1, -1, 0]),
            TernaryStrategy::new(vec![-1, 0, 1]),
            TernaryStrategy::new(vec![0, 0, 0]),
        ];
        let got = Entropy::population_diversity(&pop);
        assert!((got - 7.0 / 3.0).abs() < 1e-12, "got {got}");
    }

    /// Hand-computed Shannon entropy for [-1, 0, 0, 1, 1, 1]:
    /// p(-1)=1/6, p(0)=2/6, p(+1)=3/6.
    /// H = -(1/6 log2 1/6 + 2/6 log2 2/6 + 3/6 log2 3/6) ≈ 1.4591 bits.
    #[test]
    fn test_entropy_hand_computed_six_choices() {
        let s = TernaryStrategy::new(vec![-1, 0, 0, 1, 1, 1]);
        let p = |k: f64| -k * k.log2();
        let expected = p(1.0 / 6.0) + p(2.0 / 6.0) + p(3.0 / 6.0);
        assert!((Entropy::strategy_entropy(&s) - expected).abs() < 1e-12);
        // And it must respect the 0 ≤ H ≤ log2(3) bound.
        let h = Entropy::strategy_entropy(&s);
        assert!(h >= 0.0 && h <= Entropy::max_entropy() + 1e-12);
    }

    /// `normalized_entropy` ∈ [0, 1] for an arbitrary unbalanced strategy.
    #[test]
    fn test_normalized_entropy_in_unit_interval() {
        let s = TernaryStrategy::new(vec![-1, -1, 0, 1, 1, 1]);
        let n = Entropy::normalized_entropy(&s);
        assert!((0.0..=1.0).contains(&n));
    }
}
