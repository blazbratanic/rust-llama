use num_traits;

pub trait Numeric: num_traits::NumAssign + Clone + Default + Copy + std::fmt::Debug {}
impl<T> Numeric for T where T: num_traits::NumAssign + Clone + Default + Copy + std::fmt::Debug {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix<T: Numeric, const DIM: usize> {
    data_: std::vec::Vec<T>,
    dims_: [usize; DIM],
    contiguous_: bool,
}

fn print_matrix<T: Numeric>(
    data: &std::vec::Vec<T>,
    dims: &[usize],
    offset: usize,
    f: &mut std::fmt::Formatter,
) -> std::fmt::Result {
    if dims.len() == 1 {
        let mut row = vec![T::default(); dims[0]];
        for i in 0..dims[0] {
            row[i] = data[i + offset];
        }
        write!(f, "{:?}\n", row)?;
    } else {
        let stride: usize = dims[1..].iter().fold(1, |acc, x| acc * x);
        write!(f, "[\n")?;
        for i in 0..dims[0] {
            print_matrix::<T>(
                data,
                dims[1..].try_into().expect("Invalid"),
                offset + i * stride,
                f,
            )?;
        }
        write!(f, "]\n")?;
    }
    Ok(())
}

impl<T: Numeric, const DIM: usize> std::fmt::Display for Matrix<T, DIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        return print_matrix(self.data(), self.dims(), 0, f);
    }
}

impl<T: Numeric, const DIM: usize> Matrix<T, DIM> {
    // Constructs a new, empty `Matric<T>`.
    pub fn new(dims: &[usize; DIM]) -> Self {
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
    #[allow(dead_code)]
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
        assert!(DIM >= 2);
        Self {
            rows: m.dims()[DIM - 2],
            cols: m.dims()[DIM - 1],
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
        assert!(DIM >= 2);
        Self {
            rows: m.dims()[DIM - 2],
            cols: m.dims()[DIM - 1],
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
    assert!(
        lhs.cols == rhs.rows,
        "Invalid matrix shapes. Must be of form (n, k) (k, m)"
    );

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

pub fn matmul1d<T: Numeric>(lhs: &Matrix<T, 1>, rhs: &Matrix<T, 1>) -> T {
    return std::iter::zip(lhs.data(), rhs.data())
        .fold(T::default(), |acc, (x1, x2)| acc + *x1 * *x2);
}

// --- Matrix mul ---
pub fn matmul<T: Numeric, const DIM: usize>(
    lhs: &Matrix<T, DIM>,
    rhs: &Matrix<T, DIM>,
) -> Matrix<T, DIM> {
    for i in 0..DIM - 2 {
        assert!(lhs.dims()[i] == rhs.dims()[i], "Incompatible matrix sizes.");
    }
    let mut dims = lhs.dims().clone();
    dims[DIM - 1] = rhs.dims()[DIM - 1];

    let mut output = Matrix::<T, DIM>::new(dims);

    if DIM == 2 {
        matmul_impl(
            MatrixView2::<T>::new(&lhs, 0),
            MatrixView2::<T>::new(&rhs, 0),
            MutableMatrixView2::<T>::new(&mut output, 0),
        );
    } else {
        let lstride = lhs.dims()[DIM - 1] * lhs.dims()[DIM - 2];
        let rstride = rhs.dims()[DIM - 1] * rhs.dims()[DIM - 2];
        let ostride = output.dims()[DIM - 1] * output.dims()[DIM - 2];

        let num_rep: usize = lhs.dims().iter().fold(1, |acc, x| acc * x) / lstride;

        for d in 0..num_rep {
            matmul_impl(
                MatrixView2::<T>::new(&lhs, d * lstride),
                MatrixView2::<T>::new(&rhs, d * rstride),
                MutableMatrixView2::<T>::new(&mut output, d * ostride),
            );
        }
    }
    return output;
}
