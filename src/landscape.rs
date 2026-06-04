//! Fitness landscape topology analysis.

use crate::{Environment, FitnessEvaluator, TernaryStrategy};
use std::collections::HashMap;

/// A point in the fitness landscape.
#[derive(Clone, Debug)]
pub struct LandscapePoint {
    pub strategy: TernaryStrategy,
    pub fitness: f64,
}

impl LandscapePoint {
    fn new(strategy: TernaryStrategy, fitness: f64) -> Self {
        Self { strategy, fitness }
    }
}

/// A saddle point between multiple peaks.
#[derive(Clone, Debug)]
pub struct SaddlePoint {
    pub strategy: TernaryStrategy,
    pub fitness: f64,
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

        // Find peaks: strategies where no neighbor has higher fitness
        let peaks: Vec<LandscapePoint> = points
            .iter()
            .filter(|p| {
                let neighbors = p.strategy.neighbors();
                neighbors.iter().all(|n| {
                    let n_fitness = fitness_map.get(n.choices()).copied().unwrap_or(f64::NEG_INFINITY);
                    n_fitness <= p.fitness
                })
            })
            .cloned()
            .collect();

        let global_peak = peaks
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
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
        let peak_strategies: Vec<&TernaryStrategy> = self.peaks.iter().map(|p| &p.strategy).collect();
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
    /// steepest-ascent hill climbing.
    pub fn basin_of(&self, peak: &TernaryStrategy) -> Vec<TernaryStrategy> {
        let _peak_fitness = self
            .fitness_map
            .get(peak.choices())
            .copied()
            .unwrap_or(0.0);

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
    fn steepest_ascent_target(&self, start: &TernaryStrategy) -> TernaryStrategy {
        let mut current = start.clone();
        loop {
            let current_fitness = self
                .fitness_map
                .get(current.choices())
                .copied()
                .unwrap_or(0.0);

            let neighbors = current.neighbors();
            let best_neighbor = neighbors.iter().max_by(|a, b| {
                let fa = self.fitness_map.get(a.choices()).copied().unwrap_or(f64::NEG_INFINITY);
                let fb = self.fitness_map.get(b.choices()).copied().unwrap_or(f64::NEG_INFINITY);
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            });

            match best_neighbor {
                Some(n) => {
                    let n_fitness = self
                        .fitness_map
                        .get(n.choices())
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
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
        Environment::from_rows(&[
            [1.0, 0.5, 2.0],
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ])
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
}
