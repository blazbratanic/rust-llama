use super::matrix::Matrix;
use super::traits::Numeric;

pub fn sigmoid<T: Numeric + num_traits::Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}
