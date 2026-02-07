mod experiments;

fn main() {
    // check flags and run appropriate mode first
    let args = std::env::args();
    if args.len() > 1 {
        let arg_vec: Vec<String> = args.collect();
        match arg_vec[1].as_str() {
            "--generate-ref-tables" => {
                generate_ref_tables();
                return;
            }
            "--generate-c-tables" => {
                generate_c_tables();
                return;
            }

            "--test-suite" => {
                experiments::prior_strength_experiments();
                experiments::p_fail_experiments();
                experiments::damping_experiments();
                return;
            }
            "--ps-bias-experiments" => {
                experiments::prior_strength_experiments();
                return;
            }
            "--ps-fail-histogram" => {
                experiments::simulate_true_fail_probabilities();
            }
            "--p-fail-experiments" => {
                experiments::p_fail_experiments();
                return;
            }
            "--damping-experiments" => {
                experiments::damping_experiments();
                return;
            }
            _ => {
                println!("Unknown argument: {}", arg_vec[1]);
                return;
            }
        }
    }

    loop {
        println!("\n\nSelect mode:");
        println!("[ 1 ] Exit");
        println!("[ 2 ] Generate reference tables for k_init vs npq");
        println!("[ 3 ] Run experimental suite");
        println!("[ 4 ] Generate C tables");
        println!("[ 5 ] Prior Strength Bias Experiments");
        println!("[ 6 ] p_fail Experiments");
        println!("[ 7 ] Damping Constant Experiments");
        println!("[ 8 ] Simulate true fail probabilities for p_s experiments");

        print!("= ");
        std::io::Write::flush(&mut std::io::stdout()).expect("Failed to flush stdout");
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        match input.trim() {
            "1" => break,
            "2" => generate_ref_tables(),
            "3" => test_suite(),
            "4" => generate_c_tables(),
            "5" => experiments::prior_strength_experiments(),
            "6" => experiments::p_fail_experiments(),
            "7" => experiments::damping_experiments(),
            "8" => experiments::simulate_true_fail_probabilities(),
            _ => println!("Invalid option"),
        }
    }
}

fn generate_ref_tables() {
    // We will generate 4 tables for q .8, .9, .95, .99
    // Rows will be n = 100 .. 100,000
    // Cols will be p = 1e-2 .. 1e-8
    const NS: [usize; 6] = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];
    const PS: [f64; 8] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];
    const QS: [f64; 4] = [0.8, 0.9, 0.95, 0.99];

    for &q in QS.iter() {
        println!("\nTable for q = {q}");
        print!("n\\p\t");
        for &p in PS.iter() {
            print!("{p:.1e}\t");
        }
        println!();

        for &n in NS.iter() {
            print!("{n:0e}\t");
            for &p in PS.iter() {
                let biod = biod::State::new(
                    biod::Options {
                        confidence: q,
                        outlier_probability: p,
                        pass_ratio: 0.0,
                        timeout: None,
                    },
                    n,
                );

                let k = biod.k(biod.p_fail());

                print!("{k}\t");
            }
            println!();
        }
    }
}

fn generate_c_tables() {
    // We will generate a dense sampling of C values for k=3..=10 and q=.8-.999
    let k_bounds = (3.0, 10.0);
    let k_step = (k_bounds.1 - k_bounds.0) / 10.0;
    let q_bounds = (0.8, 0.999);
    let q_step = (q_bounds.1 - q_bounds.0) / 10.0;

    let mut c_values = Vec::new(); // (k, q, C)
    let mut k_i = 0;
    while k_bounds.0 + (k_i as f64) * k_step <= k_bounds.1 {
        let k = k_bounds.0 + (k_i as f64) * k_step;
        let mut q_i = 0;
        while q_bounds.0 + (q_i as f64) * q_step <= q_bounds.1 {
            let q = q_bounds.0 + (q_i as f64) * q_step;

            let c = k / (-(1.0f64 - q).ln());

            c_values.push((k, q, c));

            q_i += 1;
        }
        k_i += 1;
    }

    // Print in CSV format
    println!("k; q; C");
    for (k, q, c) in c_values.iter() {
        println!("{k:.4}; {q:.4}; {c:.4}");
    }
}

fn test_suite() {
    // Test 2 - for frac_survive, we use C=PI/2 to generate a floor to expected k
    // But we can do `C = desired_k_floor / (-ln(1-q))` to see values for different desired k floors
    // We do 3-10, because <3 iterations is almost meaningless, and >10 is probably too slow for extremely large sets
    // This gives us a practical limit for k_floor in terms of C
    println!("\nTest 2 - C values for various desired k floors");
    let mut c_values = Vec::new();
    for desired_k_floor in 3..=10 {
        print!("k_floor={desired_k_floor}, C =[");

        // Sample from 0.8 to 0.99 in 0.01 increments
        let mut q_i = 0;
        while 0.8 + (q_i as f64) * 0.01 <= 0.99 {
            let q = 0.8 + (q_i as f64) * 0.01;
            let c = (desired_k_floor as f64) / (-(1.0f64 - q).ln());
            c_values.push((desired_k_floor, q, c));
            q_i += 1;

            print!("q={q:.2}: {c:.4}, ");
        }

        println!("]");
    }

    // We need a value for C to produce the widest possible range of k floors in the range 3-10
    // given q in 0.8-0.99
    //
    // We can get this by taking the mean C value over all desired k floors and q values
    let all_c_sum: f64 = c_values.iter().map(|(_, _, c)| *c).sum();
    let c_mean: f64 = all_c_sum / (c_values.len() as f64);

    let variance = c_values
        .iter()
        .map(|(_, _, c)| (*c - c_mean).powi(2))
        .sum::<f64>()
        / (c_values.len() as f64);
    let std_dev = variance.sqrt();

    // We can now get C +- std_dev as a practical range for C
    let c_min = c_mean - std_dev;
    let c_max = c_mean + std_dev;
    println!("\nCalculating practical C range for desired k floors 3-10 over q .8-.99...");
    println!(
        "Practical C range for desired k floors 3-10 over q .8-.99: C mean: {c_mean:.4}, C std_dev: {std_dev:.4} (variance: {variance:.4}), C range: [{c_min:.4}, {c_max:.4}]"
    );

    // Now we can calculate the effective range of k floors for values across that C range, given common q values (0.8, 0.9, 0.95, 0.99, 0.999)
    println!("\nCalculating effective k floor range over practical C range...");
    for &q in [0.8, 0.9, 0.95, 0.99, 0.999].iter() {
        let k_floor_min = c_min * (-(1.0f64 - q).ln());
        let k_floor_mean = c_mean * (-(1.0f64 - q).ln());
        let k_floor_max = c_max * (-(1.0f64 - q).ln());

        println!(
            "q={q:.3}: k_floor range: [{k_floor_min:.4}, {k_floor_max:.4}] (mean: {k_floor_mean:.4})"
        );
    }

    // Now as a table, showing sample C as rows, and columns for common q values - cells  show k_floor
    println!("\nC vs k_floor table:");
    print!("C\\q\t");
    for &q in [0.8, 0.9, 0.95, 0.99, 0.999].iter() {
        print!("{q:.3}\t");
    }
    println!();
    let mut c_i = 0;
    while c_min + (c_i as f64) * (std_dev / 5.0) <= c_max {
        let c = c_min + (c_i as f64) * (std_dev / 5.0);
        c_i += 1;

        print!("{c:.4}\t");
        for &q in [0.8, 0.9, 0.95, 0.99, 0.999].iter() {
            let k_floor = c * (-(1.0f64 - q).ln());
            print!("{k_floor:.4}\t");
        }
        println!();
    }
    // Also show pi/2 and pi as C candidates
    for &c in [std::f64::consts::PI / 2.0, std::f64::consts::PI].iter() {
        print!("{c:.4}\t");
        for &q in [0.8, 0.9, 0.95, 0.99, 0.999].iter() {
            let k_floor = c * (-(1.0f64 - q).ln());
            print!("{k_floor:.4}\t");
        }
        println!();
    }

    // Calculate the C for which k_floor = 3.0 at q = 0.8
    // And the C for which k_floor = 10.0 at q = 0.999
    let c_for_k3_q08 = 3.0 / (-(1.0f64 - 0.8).ln());
    let c_for_k10_q0999 = 10.0 / (-(1.0f64 - 0.999).ln());
    println!(
        "\nC for k_floor=3.0 at q=0.8: {c_for_k3_q08:.4}, C for k_floor=10.0 at q=0.999: {c_for_k10_q0999:.4}"
    );
}
