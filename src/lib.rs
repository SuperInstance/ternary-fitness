//! Fitness landscape analysis for ternary agent systems {-1, 0, +1}.
//!
//! This crate provides tools for analyzing fitness landscapes over strategies
//! composed of ternary choices (-1, 0, +1), including exhaustive enumeration,
//! landscape topology analysis, Pareto fronts, and entropy measures.

#![forbid(unsafe_code)]

mod entropy;
mod environment;
mod evaluator;
mod exhaustive;
mod landscape;
mod pareto;
mod strategy;

pub use entropy::Entropy;
pub use environment::Environment;
pub use evaluator::FitnessEvaluator;
pub use exhaustive::ExhaustiveSearch;
pub use landscape::{FitnessLandscape, LandscapePoint, SaddlePoint};
pub use pareto::{Objective, ParetoFront, ParetoSolution};
pub use strategy::TernaryStrategy;
