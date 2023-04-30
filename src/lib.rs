#![cfg_attr(not(test), no_std)]

#[cfg(test)]
pub mod runtime;

mod comptime;
pub use comptime::{CMatrix, CVector, Operation};
pub use const_soft_float;
