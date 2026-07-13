//! Fitness landscape topology analysis.

use crate::{Environment, FitnessEvaluator, TernaryStrategy};
use std::collections::HashMap;

/// A point in the fitness landscape: a strategy paired with its evaluated fitness.
#[derive(Clone, Debug)]
pub struct LandscapePoint {
    /// The strategy this point refers to.
    pub strategy: TernaryStrategy,
    /// Fitness of `strategy` under the environment the landscape was built from.
    pub fitness: f64,
}

impl LandscapePoint {
    fn new(strategy: TernaryStrategy, fitness: f64) -> Self {
        Self { strategy, fitness }
    }
}

/// A saddle point: a strategy that lies on the watershed between two or more
/// distinct peaks (i.e. multiple peaks are reachable as uphill neighbors).
#[derive(Clone, Debug)]
pub struct SaddlePoint {
    /// The strategy at the saddle.
    pub strategy: TernaryStrategy,
    /// Fitness of `strategy`.
    pub fitness: f64,
    /// Distinct peaks whose basins meet at this saddle.
    pub adjacent_peaks: Vec<TernaryStrategy>,
}

/// The complete fitness landscape over all 3^n strategies.
#[derive(Clone, Debug)]
pub struct FitnessLandscape {
    points: Vec<LandscapePoint>,
    peaks: Vec<LandscapePoint>,
    global_peak: Option<LandscapePoint>,
    fitness_map: HashMap<Vec<i8>, f64>,
}

impl FitnessLandscape {
    /// Build the complete landscape for the given environment.
    pub fn build(env: &Environment) -> Self {
        let n = env.num_states();
        let total = 3usize.pow(n as u32);
        let mut points = Vec::with_capacity(total);
        let mut fitness_map = HashMap::with_capacity(total);

        // Generate all strategies and evaluate
        for i in 0..total {
            let mut choices = Vec::with_capacity(n);
            let mut idx = i;
            for _ in 0..n {
                choices.push((idx % 3) as i8 - 1);
                idx /= 3;
            }
            let strategy = TernaryStrategy::new_unchecked(choices);
            let fitness = FitnessEvaluator::evaluate(&strategy, env);
            fitness_map.insert(strategy.choices().to_vec(), fitness);
            points.push(LandscapePoint::new(strategy, fitness));
        }

        // Find peaks: strategies where no neighbor has higher fitness.
        // All neighbors of a same-length ternary strategy are themselves in
        // the landscape, so the `expect` is a programming-error assertion
        // rather than a silent fallback.
        let peaks: Vec<LandscapePoint> = points
            .iter()
            .filter(|p| {
                let neighbors = p.strategy.neighbors();
                neighbors.iter().all(|n| {
                    let n_fitness = fitness_map
                        .get(n.choices())
                        .copied()
                        .expect("neighbor must be present in landscape");
                    n_fitness <= p.fitness
                })
            })
            .cloned()
            .collect();

        let global_peak = peaks
            .iter()
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
            .cloned();

        Self {
            points,
            peaks,
            global_peak,
            fitness_map,
        }
    }

    /// Get the fitness of a specific strategy.
    pub fn fitness_of(&self, strategy: &TernaryStrategy) -> Option<f64> {
        self.fitness_map.get(strategy.choices()).copied()
    }

    /// All landscape points.
    pub fn points(&self) -> &[LandscapePoint] {
        &self.points
    }

    /// All local peaks (strict: no neighbor has higher fitness).
    pub fn peaks(&self) -> &[LandscapePoint] {
        &self.peaks
    }

    /// The global peak (highest fitness strategy).
    pub fn global_peak(&self) -> Option<&LandscapePoint> {
        self.global_peak.as_ref()
    }

    /// Find saddle points: strategies adjacent to multiple distinct peaks.
    pub fn saddle_points(&self) -> Vec<SaddlePoint> {
        let peak_strategies: Vec<&TernaryStrategy> =
            self.peaks.iter().map(|p| &p.strategy).collect();
        let mut saddles = Vec::new();

        for point in &self.points {
            let neighbors: Vec<TernaryStrategy> = point.strategy.neighbors();
            let adjacent_peak_indices: Vec<usize> = neighbors
                .iter()
                .filter_map(|n| {
                    let n_fitness = self.fitness_map.get(n.choices()).copied()?;
                    // A neighbor is a "peak direction" if it has >= fitness (climbing)
                    if n_fitness >= point.fitness {
                        if let Some(pos) = peak_strategies.iter().position(|p| *p == n) {
                            return Some(pos);
                        }
                    }
                    None
                })
                .collect();

            if adjacent_peak_indices.len() >= 2 {
                let adjacent_peaks: Vec<TernaryStrategy> = adjacent_peak_indices
                    .into_iter()
                    .map(|i| peak_strategies[i].clone())
                    .collect();
                saddles.push(SaddlePoint {
                    strategy: point.strategy.clone(),
                    fitness: point.fitness,
                    adjacent_peaks,
                });
            }
        }

        saddles
    }

    /// Find the basin of attraction for a given peak.
    ///
    /// The basin consists of all strategies that would reach this peak via
    /// steepest-ascent hill climbing. Strategies whose peak is not in the
    /// landscape return a basin containing only the peak itself.
    ///
    /// # Panics
    ///
    /// Panics if `peak` is not present in this landscape (its fitness is
    /// unknown). This is a programming error rather than a recoverable
    /// condition — the landscape must have been built from the same
    /// strategy space the peak belongs to.
    pub fn basin_of(&self, peak: &TernaryStrategy) -> Vec<TernaryStrategy> {
        // Explicit contract check: a peak that isn't in the landscape is a
        // caller bug. We avoid the silent `unwrap_or(0.0)` fallback that
        // would otherwise mask it and produce a misleading basin.
        if !self.fitness_map.contains_key(peak.choices()) {
            panic!("basin_of: peak {} is not present in this landscape", peak);
        }

        let mut basin = vec![peak.clone()];
        let mut visited = std::collections::HashSet::new();
        visited.insert(peak.choices().to_vec());

        // Work backwards: find all strategies whose steepest ascent leads to this peak
        for point in &self.points {
            if visited.contains(point.strategy.choices()) {
                continue;
            }
            if self.steepest_ascent_target(&point.strategy) == *peak {
                basin.push(point.strategy.clone());
            }
        }

        basin
    }

    /// Find which peak a strategy reaches via steepest ascent.
    ///
    /// Assumes `start` is a strategy present in this landscape. All neighbors
    /// of a same-length ternary strategy are also in the landscape, so the
    /// `expect` calls below are genuine programming-error assertions, not
    /// silent fallbacks.
    fn steepest_ascent_target(&self, start: &TernaryStrategy) -> TernaryStrategy {
        let mut current = start.clone();
        loop {
            let current_fitness = self
                .fitness_map
                .get(current.choices())
                .copied()
                .expect("steepest-ascent start must be present in landscape");

            let neighbors = current.neighbors();
            let best_neighbor = neighbors.iter().max_by(|a, b| {
                let fa = self
                    .fitness_map
                    .get(a.choices())
                    .copied()
                    .expect("neighbor must be present in landscape");
                let fb = self
                    .fitness_map
                    .get(b.choices())
                    .copied()
                    .expect("neighbor must be present in landscape");
                fa.total_cmp(&fb)
            });

            match best_neighbor {
                Some(n) => {
                    let n_fitness = self
                        .fitness_map
                        .get(n.choices())
                        .copied()
                        .expect("neighbor must be present in landscape");
                    if n_fitness > current_fitness {
                        current = n.clone();
                    } else {
                        return current;
                    }
                }
                None => return current,
            }
        }
    }

    /// Total number of strategies in the landscape.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Get the fitness range (min, max).
    pub fn fitness_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let min = self
            .points
            .iter()
            .map(|p| p.fitness)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .points
            .iter()
            .map(|p| p.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_env() -> Environment {
        Environment::from_rows(&[[1.0, 0.5, 2.0], [3.0, 1.0, 0.0], [0.0, 4.0, 1.0]])
    }

    #[test]
    fn test_landscape_size() {
        let landscape = FitnessLandscape::build(&test_env());
        assert_eq!(landscape.size(), 27);
    }

    #[test]
    fn test_global_peak() {
        let landscape = FitnessLandscape::build(&test_env());
        let peak = landscape.global_peak().unwrap();
        assert_eq!(peak.strategy.choices(), &[1, -1, 0]);
        assert_eq!(peak.fitness, 9.0);
    }

    #[test]
    fn test_fitness_range() {
        let landscape = FitnessLandscape::build(&test_env());
        let (min, max) = landscape.fitness_range().unwrap();
        assert_eq!(max, 9.0);
        assert!(min < max);
    }

    #[test]
    fn test_fitness_lookup() {
        let landscape = FitnessLandscape::build(&test_env());
        let s = TernaryStrategy::new(vec![1, -1, 0]);
        assert_eq!(landscape.fitness_of(&s), Some(9.0));
    }

    #[test]
    fn test_peaks_exist() {
        let landscape = FitnessLandscape::build(&test_env());
        assert!(!landscape.peaks().is_empty());
        // All peaks should have fitness >= their neighbors
        for peak in landscape.peaks() {
            let neighbors = peak.strategy.neighbors();
            for n in neighbors {
                let n_fit = landscape.fitness_of(&n).unwrap();
                assert!(peak.fitness >= n_fit);
            }
        }
    }

    #[test]
    fn test_basin_of_global_peak() {
        let landscape = FitnessLandscape::build(&test_env());
        let global = landscape.global_peak().unwrap();
        let basin = landscape.basin_of(&global.strategy);
        assert!(!basin.is_empty());
        // Global peak should be in its own basin
        assert!(basin.iter().any(|s| s == &global.strategy));
    }

    #[test]
    fn test_n4_landscape_81_strategies() {
        let env = Environment::from_rows(&[
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 0.5],
            [0.5, 2.5, 1.0],
            [2.0, 1.0, 3.0],
        ]);
        let landscape = FitnessLandscape::build(&env);
        assert_eq!(landscape.size(), 81);
        let peak = landscape.global_peak().unwrap();
        // state0:+1→3, state1:-1→3, state2:+1→1, state3:+1→3 = 10
        // But wait: [3.0, 1.0, 0.5] means -1→3.0, so state1:-1→3
        // [0.5, 2.5, 1.0] means 0→2.5, so state2:0→2.5
        // Best = [1,-1,0,1] = 3+3+2.5+3 = 11.5
        assert_eq!(peak.strategy.choices(), &[1, -1, 0, 1]);
    }

    /// Hand-worked saddle-point scenario. Environment `[5, 0, 5]` per state
    /// rewards any non-zero action with 5 and zero with 0 — so the four
    /// "corner" length-2 strategies `[±1, ±1]` all share the maximum fitness
    /// 10 and are all peaks. `[0, +1]` has neighbors `[-1, +1]` and
    /// `[+1, +1]`, both peaks with fitness 10 (> 5), so it must be reported
    /// as a saddle between them.
    #[test]
    fn test_saddle_point_worked_example() {
        let env = Environment::from_rows(&[[5.0, 0.0, 5.0], [5.0, 0.0, 5.0]]);
        let landscape = FitnessLandscape::build(&env);

        // Sanity: the four corners are peaks at fitness 10.
        let peak_strategies: Vec<&TernaryStrategy> =
            landscape.peaks().iter().map(|p| &p.strategy).collect();
        for corner in &[-1_i8, 1_i8] {
            for other in &[-1_i8, 1_i8] {
                let needle = TernaryStrategy::new(vec![*corner, *other]);
                let found = peak_strategies.iter().find(|s| **s == &needle);
                assert!(found.is_some(), "expected [{corner}, {other}] to be a peak");
                assert_eq!(landscape.fitness_of(&needle), Some(10.0));
            }
        }

        let saddles = landscape.saddle_points();
        assert!(!saddles.is_empty(), "expected at least one saddle");

        // [0, +1] must be a saddle adjacent to peaks [-1, +1] and [+1, +1].
        let s_target = TernaryStrategy::new(vec![0, 1]);
        let saddle = saddles
            .iter()
            .find(|s| s.strategy == s_target)
            .expect("expected [0, +1] to be a saddle");
        assert!(
            saddle
                .adjacent_peaks
                .iter()
                .any(|s| s.choices() == [-1, 1]),
            "saddle [0, +1] should be adjacent to peak [-1, +1]"
        );
        assert!(
            saddle.adjacent_peaks.iter().any(|s| s.choices() == [1, 1]),
            "saddle [0, +1] should be adjacent to peak [+1, +1]"
        );
    }

    /// NaN safety: building a landscape over an environment that contains
    /// NaN rewards must not panic. Previously the peak-detection and
    /// global-peak `max_by` used `.partial_cmp().unwrap()` which panics on
    /// NaN; the `total_cmp` rewrite should tolerate it.
    #[test]
    fn test_landscape_build_with_nan_does_not_panic() {
        let env = Environment::from_rows(&[[1.0, f64::NAN, 2.0], [3.0, 1.0, 0.0]]);
        let landscape = FitnessLandscape::build(&env);
        assert_eq!(landscape.size(), 9);
        // global_peak() still returns *something* rather than panicking.
        let _ = landscape.global_peak();
        // peaks() is internally consistent: no peak has a strictly better neighbor.
        for peak in landscape.peaks() {
            for n in peak.strategy.neighbors() {
                if let Some(nf) = landscape.fitness_of(&n) {
                    assert!(peak.fitness.total_cmp(&nf) != std::cmp::Ordering::Less);
                }
            }
        }
    }

    /// Edge case: empty environment. Landscape contains a single point (the
    /// empty strategy with fitness 0); that point is the global peak.
    #[test]
    fn test_landscape_empty_environment() {
        let env = Environment::new();
        let landscape = FitnessLandscape::build(&env);
        assert_eq!(landscape.size(), 1);
        assert!(landscape.global_peak().unwrap().strategy.is_empty());
        assert_eq!(landscape.fitness_range(), Some((0.0, 0.0)));
    }

    /// All-identical-fitness landscape (constant reward). Every strategy has
    /// fitness 1.0; every strategy is therefore a peak (no neighbor is
    /// *strictly* higher). `global_peak` returns one of them.
    #[test]
    fn test_landscape_constant_fitness() {
        let env = Environment::from_rows(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let landscape = FitnessLandscape::build(&env);
        assert_eq!(landscape.size(), 9);
        for p in landscape.points() {
            assert_eq!(p.fitness, 2.0);
        }
        // Every strategy is a peak under the non-strict definition.
        assert_eq!(landscape.peaks().len(), 9);
        assert_eq!(landscape.global_peak().unwrap().fitness, 2.0);
        assert_eq!(landscape.fitness_range(), Some((2.0, 2.0)));
    }

    /// `basin_of` must panic on a strategy not in the landscape (silent
    /// fallback was the prior behavior; now it surfaces a programming error).
    #[test]
    #[should_panic(expected = "is not present in this landscape")]
    fn test_basin_of_unknown_peak_panics() {
        let landscape = FitnessLandscape::build(&test_env());
        // Length mismatch — definitely not in the landscape.
        let bad = TernaryStrategy::new(vec![1, -1, 0, 1]);
        landscape.basin_of(&bad);
    }

    /// `basin_of(global_peak)` must contain the global peak and every
    /// strategy whose steepest-ascent walk ends at that peak. Independent
    /// re-walk here cross-checks the implementation.
    #[test]
    fn test_basin_of_global_peak_is_self_consistent() {
        let landscape = FitnessLandscape::build(&test_env());
        let global = landscape.global_peak().unwrap().strategy.clone();
        let basin = landscape.basin_of(&global);

        // The peak itself is in the basin.
        assert!(basin.iter().any(|s| s == &global));

        // Every basin member's steepest-ascent walk (computed independently
        // below) must end at `global`.
        for member in &basin {
            let mut cur = member.clone();
            for _ in 0..32 {
                let cur_fit = landscape.fitness_of(&cur).unwrap();
                let next = cur
                    .neighbors()
                    .into_iter()
                    .filter(|n| landscape.fitness_of(n).is_some())
                    .max_by(|a, b| {
                        landscape
                            .fitness_of(a)
                            .unwrap()
                            .total_cmp(&landscape.fitness_of(b).unwrap())
                    })
                    .unwrap();
                if landscape.fitness_of(&next).unwrap() > cur_fit {
                    cur = next;
                } else {
                    break;
                }
            }
            assert_eq!(cur, global, "basin member did not climb to global peak");
        }
    }
}
