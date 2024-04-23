pub trait Numeric: num_traits::NumAssign + Clone + Default + Copy + std::fmt::Debug {}
impl<T> Numeric for T where T: num_traits::NumAssign + Clone + Default + Copy + std::fmt::Debug {}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Dims {
    // Max num of dims is 32.
    dims_: [usize; 32],
    ndims_: usize,
}

impl Dims {
    pub fn dims(&self) -> &[usize] {
        &self.dims_[..self.ndims_]
    }
    pub fn set_dim_size(&mut self, dim: usize, size: usize) {
        self.dims_[dim] = size;
    }
}

impl Into<Dims> for &[usize] {
    fn into(self) -> Dims {
        assert!(self.len() <= 32, "Max number of dimensions is 32");
        assert!(
            self.len() > 0,
            "Cannot create a 0 dimensional Dim structure"
        );
        let mut d = [0; 32];
        for i in 0..self.len() {
            d[i] = self[i];
        }

        Dims {
            dims_: d,
            ndims_: self.len(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix<T: Numeric> {
    data_: std::vec::Vec<T>,
    dims_: Dims,
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

impl<T: Numeric> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Dims {:?}\n", self.dims())?;
        return print_matrix(self.data(), self.dims(), 0, f);
    }
}

impl<T: Numeric> Matrix<T> {
    // Constructs a new, empty `Matric<T>`.
    pub fn new(dims: &[usize]) -> Self {
        Self::with_value(dims, T::default())
    }
    // Constructs a new, empty `Matric<T>`.
    pub fn with_value(dims: &[usize], value: T) -> Self {
        assert!(dims.len() > 0, "Matrix dim must be > 0");
        let size: usize = dims.iter().fold(1, |acc, x| acc * x);
        Self {
            data_: vec![value; size],
            dims_: dims.into(),
            contiguous_: true,
        }
    }

    pub fn dims(&self) -> &[usize] {
        self.dims_.dims()
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

// impl<T: Default + Clone>
pub fn equal_dims<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>)
where
    T: Default,
{
    assert!(
        m1.dims().len() == m2.dims().len(),
        "Matrices have different number of dimensions"
    );

    for (d1, d2) in std::iter::zip(m1.dims().iter().rev(), m2.dims().iter().rev()) {
        assert!(d1 == d2, "Dimension size mismatch.");
    }
}

// --- Scalar ---
// Add scalar
impl<T: Numeric> core::ops::AddAssign<T> for Matrix<T> {
    fn add_assign(self: &mut Matrix<T>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] += rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Add)]
pub fn add<T: Numeric>(self: Matrix<T>, rhs: T) -> Matrix<T> {
    let mut result = self.clone();
    result += rhs;
    return result;
}

// Mul scalar
impl<T: Numeric> core::ops::MulAssign<T> for Matrix<T> {
    fn mul_assign(self: &mut Matrix<T>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] *= rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Mul)]
pub fn mul<T: Numeric>(self: Matrix<T>, rhs: T) -> Matrix<T> {
    let mut result = self.clone();
    result *= rhs;
    return result;
}

// Div scalar
impl<T: Numeric> core::ops::DivAssign<T> for Matrix<T> {
    fn div_assign(self: &mut Matrix<T>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] /= rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Div)]
pub fn div<T: Numeric>(self: Matrix<T>, rhs: T) -> Matrix<T> {
    let mut result = self.clone();
    result /= rhs;
    return result;
}

// Sub scalar
impl<T: Numeric> core::ops::SubAssign<T> for Matrix<T> {
    fn sub_assign(self: &mut Matrix<T>, rhs: T) {
        for i in 0..self.data_.len() {
            self.data_[i] -= rhs
        }
    }
}
#[opimps::impl_ops_rprim(core::ops::Sub)]
pub fn sub<T: Numeric>(self: Matrix<T>, rhs: T) -> Matrix<T> {
    let mut result = self.clone();
    result -= rhs;
    return result;
}

// --- Element wise ---
// Add matrix
#[opimps::impl_ops_assign(std::ops::AddAssign)]
pub fn add_assign<T: Numeric>(self: Matrix<T>, rhs: Matrix<T>) {
    equal_dims(&self, &rhs);

    for i in 0..self.data_.len() {
        self.data_[i] += rhs.data_[i];
    }
}

#[opimps::impl_ops(core::ops::Add)]
pub fn add<T: Numeric>(self: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    equal_dims(&self, &rhs);
    let mut result = self.clone();
    result += rhs;
    return result;
}

// Sub matrix
#[opimps::impl_ops_assign(std::ops::SubAssign)]
pub fn sub_assign<T: Numeric>(self: Matrix<T>, rhs: Matrix<T>) {
    equal_dims(&self, &rhs);
    for i in 0..self.data_.len() {
        self.data_[i] -= rhs.data_[i];
    }
}
#[opimps::impl_ops(core::ops::Sub)]
pub fn sub<T: Numeric>(self: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    equal_dims(&self, &rhs);
    let mut result = self.clone();
    result -= rhs;
    return result;
}

// Mul matrix
#[opimps::impl_ops_assign(std::ops::MulAssign)]
pub fn mul_assign<T: Numeric>(self: Matrix<T>, rhs: Matrix<T>) {
    equal_dims(&self, &rhs);
    for i in 0..self.data_.len() {
        self.data_[i] *= rhs.data_[i];
    }
}
#[opimps::impl_ops(core::ops::Mul)]
pub fn mul<T: Numeric>(self: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
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
    pub fn new(m: &'a mut Matrix<T>, _offset: usize) -> MutableMatrixView2<'a, T> {
        assert!(m.dims().len() >= 2);
        Self {
            rows: m.dims()[m.dims().len() - 2],
            cols: m.dims()[m.dims().len() - 1],
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
    pub fn new(m: &'a Matrix<T>, _offset: usize) -> MatrixView2<'a, T> {
        assert!(m.dims().len() >= 2);
        Self {
            rows: m.dims()[m.dims().len() - 2],
            cols: m.dims()[m.dims().len() - 1],
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

// --- Matrix mul ---

pub fn matmul_assign<T: Numeric>(lhs: &Matrix<T>, rhs: &Matrix<T>, mut output: &mut Matrix<T>) {
    let lhs_ndims: usize = lhs.dims().len();
    for i in 0..lhs_ndims - 2 {
        assert!(lhs.dims()[i] == rhs.dims()[i], "Incompatible matrix sizes.");
    }
    let mut dims: Dims = lhs.dims().into();
    dims.set_dim_size(lhs_ndims - 1, rhs.dims()[lhs_ndims - 1]);

    assert!(
        output.dims() == dims.dims(),
        "Output matrix has invalid size.",
    );

    if lhs_ndims == 2 {
        matmul_impl(
            MatrixView2::<T>::new(&lhs, 0),
            MatrixView2::<T>::new(&rhs, 0),
            MutableMatrixView2::<T>::new(&mut output, 0),
        );
    } else {
        let lstride = lhs.dims()[lhs_ndims - 1] * lhs.dims()[lhs_ndims - 2];
        let rstride = rhs.dims()[lhs_ndims - 1] * rhs.dims()[lhs_ndims - 2];
        let ostride = output.dims()[lhs_ndims - 1] * output.dims()[lhs_ndims - 2];

        let num_rep: usize = lhs.dims()[..lhs_ndims - 2].iter().fold(1, |acc, x| acc * x);

        for d in 0..num_rep {
            matmul_impl(
                MatrixView2::<T>::new(&lhs, d * lstride),
                MatrixView2::<T>::new(&rhs, d * rstride),
                MutableMatrixView2::<T>::new(&mut output, d * ostride),
            );
        }
    }
}

pub fn matmul<T: Numeric>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
    let lhs_ndims: usize = lhs.dims().len();
    for i in 0..lhs_ndims - 2 {
        assert!(lhs.dims()[i] == rhs.dims()[i], "Incompatible matrix sizes.");
    }
    let mut dims: Dims = lhs.dims().into();
    dims.set_dim_size(lhs_ndims - 1, rhs.dims()[lhs_ndims - 1]);

    let mut output = Matrix::<T>::new(dims.dims());
    matmul_assign(lhs, rhs, &mut output);
    return output;
}
