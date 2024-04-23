use super::matrix::Matrix;
use super::traits::Numeric;

pub trait NNModule {
    type Input;
    type Output;
    fn forward(input: &Self::Input) -> &Self::Output;
}

struct MultiheadAttention<T: Numeric> {
    d_model: usize,
    num_heads: usize,
    d_k: usize,
    w_q: Matrix<T>,
    w_k: Matrix<T>,
    w_v: Matrix<T>,
    w_o: Matrix<T>,
}

impl<T: Numeric> MultiheadAttention<T> {
    pub fn scaled_dot_product_attention(
        &self,
        q: &Matrix<T>,
        k: &Matrix<T>,
        v: &Matrix<T>,
        mask: Option<&Matrix<T>>,
    ) {

    }
}
