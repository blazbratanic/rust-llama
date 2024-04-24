pub trait Numeric:
    num_traits::NumAssign + Clone + Default + Copy + std::fmt::Debug
{
}
impl<T> Numeric for T where
    T: num_traits::NumAssign + Clone + Default + Copy + std::fmt::Debug
{
}

pub trait One {
    type Item;
    fn one() -> Self::Item;
}
impl One for f32 {
    type Item = f32;
    fn one() -> f32 {
        1.0f32
    }
}

impl One for f64 {
    type Item = f64;
    fn one() -> f64 {
        1.0f64
    }
}
