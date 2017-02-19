#![feature(collections)]
//#![no_std]
extern crate core;


#[macro_use]
extern crate collections;

extern crate array;
extern crate one;
extern crate zero;


mod ndarray;


pub use ndarray::NDArray;
