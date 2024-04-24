// Assume batch size == 1 for now.
//
use super::matrix::matmul;
use super::matrix::Matrix;
use super::activations::softmax_last_dim;

struct MultiheadAttention {
    d_model: usize,
    num_heads: usize,
    d_k: usize,
    w_q: Matrix<f32>,
    w_k: Matrix<f32>,
    w_v: Matrix<f32>,
    w_o: Matrix<f32>,
}

impl MultiheadAttention {
    pub fn scaled_dot_product_attention(
        &self,
        q: &Matrix<f32>,
        k: &Matrix<f32>,
        v: &Matrix<f32>,
        mask: Option<&Matrix<i8>>,
    ) -> &Matrix<f32> {
        let mut attn_scores = matmul(q, k.transpose(-2, -1)) * (1.0f32 / (self.d_k as f32).sqrt());

        if mask.is_some() {
            attn_scores.fill_where(mask.unwrap(), 0, -1e9f32);

        softmax_last_dim(&mut attn_scores);

        matmul(&attn_scores, v);
        }

        return &self.w_q;
    }

    pub fn split_heads(&self, x: &Matrix<f32>) -> &Matrix<f32> {
        assert!(x.dims().len() == 2, "Expecting (seq_len, d_model) matrix");

        return &self.w_q;
    }

    pub fn combine_heads(&self, x: &Matrix<f32>) -> &Matrix<f32> {
        return &self.w_q;
    }

    fn forward(
        &self,
        q: &Matrix<f32>,
        k: &Matrix<f32>,
        v: &Matrix<f32>,
        mask: Option<&Matrix<i8>>,
    ) -> Matrix<f32> {
        let q_n: &Matrix<f32> = self.split_heads(&matmul(&self.w_q, q));
        let k_n: &Matrix<f32> = self.split_heads(&matmul(&self.w_k, k));
        let v_n: &Matrix<f32> = self.split_heads(&matmul(&self.w_v, v));

        let attn_output = self.scaled_dot_product_attention(q_n, k_n, v_n, mask);
        let output = matmul(&self.w_o, self.combine_heads(attn_output));

        return output;
    }
}
