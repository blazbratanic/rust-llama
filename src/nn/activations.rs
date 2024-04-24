use super::matrix::Matrix;
use super::traits::Numeric;

fn get_data_size(dims: &[usize]) -> usize {
    dims.iter().fold(1, |acc, x| acc * x)
}

pub fn sigmoid<T: Numeric + num_traits::Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

pub fn softmax<T: Numeric + num_traits::Float>(v: &mut std::vec::Vec<T>, offset:usize, size: usize) {
    let mut norm = T::default();

    for i in 0..size {
        norm += v[offset + i].exp();
    }
    norm = T::one() / norm;

    for i in 0..size {
        v[offset + i] *= norm;
    }
}

pub fn softmax_last_dim<T: Numeric + num_traits::Float>(m: &mut Matrix<T>) {

    let stride = m.dims().last().unwrap().clone();
    let num_iter = get_data_size(m.dims()) / stride;
    for i in 0..num_iter {
        softmax(&mut m.data(), i * stride, stride)
    }
}
