use rand::{Rng, thread_rng};

use ndarray::*;
use ndarray_linalg::*;
use ndarray::prelude::*;

use std::env;

fn main() {

    ////////////////////////////////////////////////////////////////////////////
    // random uniform sampling of d-dimensional hypercube

    let args: Vec<String> = env::args().collect();

    let samples = 1024;
    let dimensions = 512;

    let mut X = Array::ones((dimensions, 2).f());
    let mut Y = Array1::<f64>::zeros(dimensions);

    let mut tmp: f64 = 0.;
    let mut current_rmse: f64 = 0.;
    let mut p1: f64;
    let mut p2: f64;

    let mut die = thread_rng();

    for d in 1..dimensions {

        for i in 0..samples {

            for j in 0..d {
                tmp += (die.gen_range(0., 1.)-die.gen_range(0., 1.)).powi(2);
            }
            current_rmse += tmp.powf(0.5);
            tmp = 0.;
        }

        X[[d, 0]] = (d as f64).ln();
        Y[d] = current_rmse/(samples as f64);

        current_rmse = 0.;
    }


    ////////////////////////////////////////////////////////////////////////////
    // fit logarithmic curve

    // closed form solution
    let ab = (X.t().dot(&X)).inv().unwrap().dot(&X.t()).dot(&Y); 
    let a = ab[0];
    let b = ab[1];

    // compute goodness of fit using R^2 value
    let mean_y: f64 = Y.iter().sum();
    let mut u: f64 = 0.;
    let mut l: f64 = 0.;
    for i in 0..(dimensions as usize) {
        u += (Y[i]-(a+(b*X[[i, 0]]))).powf(2.);
        l += (Y[i]-mean_y).powf(2.);
    }
    let r_squared = 1. - (u/l);

    print!("R² = {:.6}\n", r_squared);
    print!("y = {:.2} ln(x) + {:.2}\n", a, b);
}