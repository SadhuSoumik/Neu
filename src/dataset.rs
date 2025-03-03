use nalgebra::DMatrix;
use std::error::Error;
use csv::ReaderBuilder;

/// Loads a dataset from a CSV file.  
/// Each row is expected to contain 784 feature values followed by 1 label (an integer 0–9).
pub fn load_dataset(file_path: &str) -> Result<(Vec<DMatrix<f64>>, Vec<DMatrix<f64>>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for result in rdr.records() {
        let record = result?;
        // Parse the first 784 values as features.
        let input_vals: Vec<f64> = record
            .iter()
            .take(784)
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        let input = DMatrix::from_row_slice(1, 784, &input_vals);
        // The 785th value is the label.
        let label: usize = record.get(784).unwrap().parse().unwrap_or(0);
        let mut target_vals = vec![0.0; 10];
        target_vals[label] = 1.0;
        let target = DMatrix::from_row_slice(1, 10, &target_vals);
        inputs.push(input);
        targets.push(target);
    }
    Ok((inputs, targets))
}

/// Generates simulated MNIST-like data if no external dataset is available.
pub fn generate_mnist_like_data(samples: usize) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..samples {
        let data: Vec<f64> = (0..784).map(|_| rng.gen_range(0.0..1.0)).collect();
        let input = DMatrix::from_row_slice(1, 784, &data);
        let label = rng.gen_range(0..10);
        let mut target_data = vec![0.0; 10];
        target_data[label] = 1.0;
        let target = DMatrix::from_row_slice(1, 10, &target_data);
        inputs.push(input);
        targets.push(target);
    }
    (inputs, targets)
}
