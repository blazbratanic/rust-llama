use num_traits;

pub trait Numeric: num_traits::NumAssign + Clone + Default + Copy {}
impl<T> Numeric for T where T: num_traits::NumAssign + Clone + Default + Copy {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix<T: Numeric, const DIM: usize> {
    data_: std::vec::Vec<T>,
    dims_: [usize; DIM],
}

impl<T: Numeric, const DIM: usize> Matrix<T, DIM> {
    // Constructs a new, empty `Matric<T>`.
    pub fn new(dims: [usize; DIM]) -> Self {
        Self::with_value(dims, T::default())
    }
    // Constructs a new, empty `Matric<T>`.
    pub fn with_value(dims: [usize; DIM], value: T) -> Self {
        let mut size: usize = 1;
        for d in dims {
            size *= d;
        }

        Self {
            data_: vec![value; size],
            dims_: dims,
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
}

// impl<T: Default + Clone, const DIM: usize>
pub fn equal_dims<T: Numeric, const DIM: usize>(m1: &Matrix<T, DIM>, m2: &Matrix<T, DIM>)
where
    T: Default,
{
    for (d1, d2) in std::iter::zip(m1.dims(), m2.dims()) {
        if d1 != d2 {
            panic!("Can only add matrices of same shape.");
        }
    }
}

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

// Add matrix
#[opimps::impl_ops_assign(std::ops::AddAssign)]
pub fn add_assign<T: Numeric, const DIM: usize>(self: Matrix<T, DIM>, rhs: Matrix<T, DIM>) {
    equal_dims(&self, &rhs);
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
        self.data_[i] -= rhs.data_[i];
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
