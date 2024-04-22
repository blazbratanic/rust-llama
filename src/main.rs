mod math;

fn main() {
    let ref m1 = math::Matrix::<f32, 3>::new([1, 2, 3]);
    let ref m2 = math::Matrix::<f32, 3>::with_value([1, 2, 3], 5.0);
    let ref m3 = math::Matrix::<f32, 3>::with_value([1, 2, 3], 2.0);
    println!("{:?}", m1);
    println!("{:?}", m2);
    println!("{:?}", m1 + m2);
    println!("{:?}", m1 + m2 + m3);
    println!("{:?}", m1 + m2 + m3 + 10.0);
    println!("{:?}", m3 * 5.0);
    println!("{:?}", (m3 * 5.0 + 9.0) / 2.0);
}
