mod math;

fn main() {
    let ref m1 = math::Matrix::<f32>::new(&[1, 2, 3]);
    let ref m2 = math::Matrix::<f32>::with_value(&[1, 2, 3], 5.0);
    let ref m3 = math::Matrix::<f32>::with_value(&[1, 2, 3], 2.0);
    println!("{}", m1);
    println!("{}", m2);
    println!("{}", m1 + m2);
    println!("{}", m1 + m2 + m3);
    println!("{}", m1 + m2 + m3 + 10.0);
    println!("{}", m3 * 5.0);
    println!("{}", (m3 * 5.0 + 9.0) / 2.0);

    let ref m4 = math::Matrix::<f32>::with_value(&[8, 4], 8.0);
    let ref m5 = math::Matrix::<f32>::with_value(&[4, 9], 5.0);
    println!("{:?}", math::matmul(m4, m5));

    let ref m6 = math::Matrix::<f32>::with_value(&[2, 8, 4], 8.0);
    let ref m7 = math::Matrix::<f32>::with_value(&[2, 4, 9], 5.0);
    println!("{}", math::matmul(m6, m7));
}
