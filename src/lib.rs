#![cfg_attr(not(test), no_std)]

#[cfg(test)]
pub mod runtime;

mod comptime;
pub use comptime::CMatrix;
