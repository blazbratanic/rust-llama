use num_traits;

pub trait Numeric: num_traits::NumAssign + Clone + Default + Copy {}
impl<T> Numeric for T where T: num_traits::NumAssign + Clone + Default + Copy {}

trait MaxSize {
    const MAX_SIZE: usize;
}

struct MaxSizeCondition<const LDIM: usize, const RDIM: usize, const CONDITION: bool> {}

impl<const LDIM: usize, const RDIM: usize> MaxSize for MaxSizeCondition<LDIM, RDIM, true> {
    const MAX_SIZE: usize = LDIM;
}

impl<const LDIM: usize, const RDIM: usize> MaxSize for MaxSizeCondition<LDIM, RDIM, false> {
    const MAX_SIZE: usize = RDIM;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix<T: Numeric, const DIM: usize> {
    data_: std::vec::Vec<T>,
    dims_: [usize; DIM],
    contiguous_: bool,
}

impl<T: Numeric, const DIM: usize> Matrix<T, DIM> {
    // Constructs a new, empty `Matric<T>`.
    pub fn new(dims: [usize; DIM]) -> Self {
        Self::with_value(dims, T::default())
    }
    // Constructs a new, empty `Matric<T>`.
    pub fn with_value(dims: [usize; DIM], value: T) -> Self {
        assert!(DIM > 0, "Matrix dim must be > 0");
        let size: usize = dims.iter().fold(1, |acc, x| acc * x);
        Self {
            data_: vec![value; size],
            dims_: dims,
            contiguous_: true,
        }
    }

    pub fn dims(&self) -> &[usize; DIM] {
        &self.dims_
    }
    pub fn data(&self) -> &std::vec::Vec<T> {
        &self.data_
    }
    pub fn mutable_data(&mut self) -> &mut std::vec::Vec<T> {
        &mut self.data_
    }
    pub fn contiguous(&self) -> bool {
        self.contiguous_
    }
}

// impl<T: Default + Clone, const DIM: usize>
pub fn equal_dims<T: Numeric, const LDIM: usize, const RDIM: usize>(
    m1: &Matrix<T, LDIM>,
    m2: &Matrix<T, RDIM>,
) where
    T: Default,
{
    for (d1, d2) in std::iter::zip(m1.dims().iter().rev(), m2.dims().iter().rev()) {
        assert!(d1 == d2, "Can only add matrices of same shape.");
    }
}

// --- Scalar ---
// Add scalar
impl<T: Numeric, const DIM: usize> core::ops::AddAssign<T> for Matrix<T, DIM> {
    fn add_assign(self: &mut Matrix<T, DIM>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] += rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Add)]
pub fn add<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: T) -> Matrix<T, DIM> {
    let mut result = self.clone();
    result += rhs;
    return result;
}

// Mul scalar
impl<T: Numeric, const DIM: usize> core::ops::MulAssign<T> for Matrix<T, DIM> {
    fn mul_assign(self: &mut Matrix<T, DIM>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] *= rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Mul)]
pub fn mul<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: T) -> Matrix<T, DIM> {
    let mut result = self.clone();
    result *= rhs;
    return result;
}

// Div scalar
impl<T: Numeric, const DIM: usize> core::ops::DivAssign<T> for Matrix<T, DIM> {
    fn div_assign(self: &mut Matrix<T, DIM>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] /= rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Div)]
pub fn div<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: T) -> Matrix<T, DIM> {
    let mut result = self.clone();
    result /= rhs;
    return result;
}

// Sub scalar
impl<T: Numeric, const DIM: usize> core::ops::SubAssign<T> for Matrix<T, DIM> {
    fn sub_assign(self: &mut Matrix<T, DIM>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] -= rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Sub)]
pub fn sub<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: T) -> Matrix<T, DIM> {
    let mut result = self.clone();
    result -= rhs;
    return result;
}

// --- Element wise ---
// Add matrix
#[opimps::impl_ops_assign(std::ops::AddAssign)]
pub fn add_assign<T: Numeric, const LDIM: usize, const RDIM: usize>(
    self: Matrix<T, LDIM>,
    rhs: Matrix<T, RDIM>,
) {
    equal_dims(&self, &rhs);
    assert!(LDIM >= RDIM, "LDIM < RDIM");

    for i in 0..self.data_.len() {
        self.data_[i] += rhs.data_[i];
    }
}

#[opimps::impl_ops(core::ops::Add)]
pub fn add<T: Numeric, const DIM: usize>(
    self: Matrix<T, DIM>,
    rhs: Matrix<T, DIM>,
) -> Matrix<T, DIM> {
    equal_dims(&self, &rhs);
    let mut result = self.clone();
    result += rhs;
    return result;
}

// Sub matrix
#[opimps::impl_ops_assign(std::ops::SubAssign)]
pub fn sub_assign<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: Matrix<T, DIM>) {
    equal_dims(&self, &rhs);
    for i in 0..self.data_.len() {
        self.data_[i] -= rhs.data_[i];
    }
}
#[opimps::impl_ops(core::ops::Sub)]
pub fn sub<T: Numeric, const DIM: usize>(
    self: Matrix<T, DIM>,
    rhs: Matrix<T, DIM>,
) -> Matrix<T, DIM> {
    equal_dims(&self, &rhs);
    let mut result = self.clone();
    result -= rhs;
    return result;
}

// Mul matrix
#[opimps::impl_ops_assign(std::ops::MulAssign)]
pub fn mul_assign<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: Matrix<T, DIM>) {
    equal_dims(&self, &rhs);
    for i in 0..self.data_.len() {
        self.data_[i] *= rhs.data_[i];
    }
}
#[opimps::impl_ops(core::ops::Mul)]
pub fn mul<T: Numeric, const DIM: usize>(
    self: Matrix<T, DIM>,
    rhs: Matrix<T, DIM>,
) -> Matrix<T, DIM> {
    equal_dims(&self, &rhs);
    let mut result = self.clone();
    result *= rhs;
    return result;
}

struct MutableMatrixView2<'a, T: Numeric> {
    rows: usize,
    cols: usize,
    offset: usize,
    data: &'a mut std::vec::Vec<T>,
}

impl<'a, T: Numeric> MutableMatrixView2<'a, T> {
    // Constructs a new, empty `Matric<T>`.
    pub fn new<const DIM: usize>(
        m: &'a mut Matrix<T, DIM>,
        _offset: usize,
    ) -> MutableMatrixView2<'a, T> {
        assert!(DIM == 2);
        Self {
            rows: m.dims()[0],
            cols: m.dims()[1],
            offset: _offset,
            data: m.mutable_data(),
        }
    }
}

struct MatrixView2<'a, T: Numeric> {
    rows: usize,
    cols: usize,
    offset: usize,
    data: &'a std::vec::Vec<T>,
}

impl<'a, T: Numeric> MatrixView2<'a, T> {
    // Constructs a new, empty `Matric<T>`.
    pub fn new<const DIM: usize>(m: &'a Matrix<T, DIM>, _offset: usize) -> MatrixView2<'a, T> {
        assert!(DIM == 2);
        Self {
            rows: m.dims()[0],
            cols: m.dims()[1],
            offset: _offset,
            data: m.data(),
        }
    }
}

fn matmul_impl<T: Numeric>(
    lhs: MatrixView2<T>,
    rhs: MatrixView2<T>,
    output: MutableMatrixView2<T>,
) {
    assert!(lhs.cols == rhs.rows, "Invalid matrix shapes. Must be of form (n, k) (k, m)");

    // (n, m) * (m, k) => (n, k)
    for lhs_r in 0..lhs.rows {
        let lhs_r_idx = lhs.offset + lhs_r * lhs.cols;

        for rhs_c in 0..rhs.cols {
            let mut buf = T::default();
            let rhs_c_idx = rhs.offset + rhs_c;

            for lhs_c in 0..lhs.cols {
                buf += lhs.data[lhs_r_idx + lhs_c] * rhs.data[lhs_c * rhs.cols + rhs_c_idx];
            }
            output.data[output.offset + lhs_r * output.cols + rhs_c] = buf;
        }
    }
}

// --- Matrix mul ---
pub fn matmul<T: Numeric, const DIM: usize>
(lhs: &Matrix<T, DIM>, rhs: &Matrix<T, DIM>) -> Matrix<T, DIM> {
    // if RDIM > LDIM {
    //     return matmul(rhs, lhs);
    // }

    if DIM == 2 {
    let mut output = Matrix::<T, 2>::new([lhs.dims()[0], rhs.dims()[1]]);
    matmul_impl(
        MatrixView2::<T>::new(&lhs, 0),
        MatrixView2::<T>::new(&rhs, 0),
        MutableMatrixView2::<T>::new(&mut output, 0),
    );
    return output;
    }
    panic!("Invalid");
}
