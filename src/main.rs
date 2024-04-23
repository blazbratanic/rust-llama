mod nn;

fn main() {
    let ref m1 = nn::Matrix::<f32>::new(&[1, 2, 3]);
    let ref m2 = nn::Matrix::<f32>::with_value(&[1, 2, 3], 5.0);
    let ref m3 = nn::Matrix::<f32>::with_value(&[1, 2, 3], 2.0);
    println!("{}", m1);
    println!("{}", m2);
    println!("{}", m1 + m2);
    println!("{}", m1 + m2 + m3);
    println!("{}", m1 + m2 + m3 + 10.0);
    println!("{}", m3 * 5.0);
    println!("{}", (m3 * 5.0 + 9.0) / 2.0);

    let ref m4 = nn::Matrix::<f32>::with_value(&[8, 4], 8.0);
    let ref m5 = nn::Matrix::<f32>::with_value(&[4, 9], 5.0);
    println!("{:?}", nn::matmul(m4, m5));

    let ref m6 = nn::Matrix::<f32>::with_value(&[2, 8, 4], 8.0);
    let ref mut m7 = nn::Matrix::<f32>::with_value(&[2, 4, 9], 1.0);
    println!("{}", nn::matmul(m6, m7));
    println!("{}", m7.apply(&nn::sigmoid::<f32>));
}
