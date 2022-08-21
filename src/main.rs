#![feature(const_eval_select)]
#![feature(const_for)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_heap)]
#![feature(const_mut_refs)]
#![feature(const_slice_from_raw_parts_mut)]
#![feature(const_swap)]
#![feature(core_intrinsics)]
#![feature(decl_macro)]
#![feature(inline_const)]
#![feature(new_uninit)]
#![feature(const_eval_limit)]
#![const_eval_limit = "0"]

use std::mem::MaybeUninit;

macro const_for(for $i:ident in ($s:expr, $e:expr) $b:block) {{
    let mut $i = $s;
    while $i < $e {
        $b;
        $i += 1;
    }
}}

// Computes error.
const fn err(a: f64, b: f64) -> f64 {
    0.5 * (a - b) * (a - b)
}

// Returns partial derivative of error function.
const fn pderr(a: f64, b: f64) -> f64 {
    a - b
}

const fn abs(n: f64) -> f64 {
    if n > 0.0 {
        n
    } else {
        -n
    }
}

const fn powu(f: f64, n: u64) -> f64 {
    match n {
        0 => panic!(),
        1 => f,
        n => f * powu(f, n - 1),
    }
}

// Computes total error of target to output.
const fn toterr(target: &[f64], output: &[f64]) -> f64 {
    let mut sum = 0.0;

    const_for!(for i in (0, target.len()) {
        sum += err(target[i], output[i]);
    });

    sum
}

// Activation function.
const fn act(x: f64) -> f64 {
    // sigmoid
    //1.0 / (1.0 + (-x).exp())

    // softsign
    x / (1.0 + abs(x))
}

// Returns partial derivative of activation function.
const fn pdact(a: f64) -> f64 {
    // sigmoid
    //a * (1.0 - a)

    // softsign
    let x = -a / (a - 1.0); // inverse of softsign
    1.0 / powu(1.0 + abs(x), 2)
}

// Returns floating point random from 0.0 - 1.0.
const fn frand(seed: u64) -> (u64, f64) {
    let a = 16807;
    let m = 2147483647;
    let seed = (a * seed) % m;
    let random = seed as f64 / m as f64;
    (seed, random)
}

// Randomizes tinn weights and biases.
const fn wbrand(t: &mut Tinn) {
    let mut seed = 981635986311532;

    const_for!(for i in (0, t.weights1.len()) {
        let (ns, r) = frand(seed);
        seed = ns;
        t.weights1[i] = r - 0.5;
    });

    const_for!(for i in (0, t.biases.len()) {
        let (ns, r) = frand(seed);
        seed = ns;
        t.biases[i] = r - 0.5;
    });
}

struct Tinn {
    // Number of inputs.
    input_size: usize,
    // All the weights.
    weights1: &'static mut [f64],
    // Hidden to output layer weights.
    weights2: &'static mut [f64],
    // Biases.
    // Number of biases is always two - Tinn only supports a single hidden layer.
    biases: &'static mut [f64],
    // Hidden layer.
    hidden: &'static mut [f64],
    // Output layer.
    output: &'static mut [f64],
}

const unsafe fn allocate_array(size: usize) -> &'static mut [f64] {
    const fn ct(size: usize) -> &'static mut [f64] {
        unsafe {
            core::slice::from_raw_parts_mut(
                core::intrinsics::const_allocate(
                    core::mem::size_of::<f64>() * size,
                    core::mem::align_of::<f64>(),
                )
                .cast::<f64>(),
                size,
            )
        }
    }

    fn rt(size: usize) -> &'static mut [f64] {
        let ptr = Box::into_raw(Box::<[MaybeUninit<f64>]>::new_zeroed_slice(size)).cast::<f64>();
        unsafe { core::slice::from_raw_parts_mut(ptr, size) }
    }

    core::intrinsics::const_eval_select((size,), ct, rt)
}

impl Tinn {
    // Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
    pub const fn build(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weights1_size = hidden_size * (input_size + output_size);
        let weights1 = unsafe { allocate_array(weights1_size) };
        let weights2 = unsafe {
            let skip = hidden_size * input_size;
            core::slice::from_raw_parts_mut(weights1.as_mut_ptr().add(skip), weights1_size - skip)
        };
        let biases_size = 2;
        let mut this = Self {
            input_size,
            weights1,
            weights2,
            biases: unsafe { allocate_array(biases_size) },
            hidden: unsafe { allocate_array(hidden_size) },
            output: unsafe { allocate_array(output_size) },
        };
        wbrand(&mut this);
        this
    }
}

// Performs back propagation.
const fn bprop(t: &mut Tinn, input: &[f64], target: &[f64], rate: f64) {
    const_for!(for i in (0, t.hidden.len()) {
        let mut sum = 0.0;
        const_for!(for j in (0, t.output.len()) {
            let a = pderr(t.output[j], target[j]);
            let b = pdact(t.output[j]);
            sum += a * b * t.weights2[j * t.hidden.len() + i];
            // Correct weights in hidden to output layer.
            t.weights2[j * t.hidden.len() + i] -= rate * a * b * t.hidden[i];
        });

        // Correct weights in input to hidden layer.
        const_for!(for j in (0, t.input_size) {
            t.weights1[i * t.input_size + j] -= rate * sum * pdact(t.hidden[i]) * input[j];
        });
    })
}

// Performs forward propagation.
const fn fprop(t: &mut Tinn, input: &[f64]) {
    // Calculate hidden layer neuron values.
    const_for!(for i in (0, t.hidden.len()) {
        let mut sum = 0.0;
        const_for!(for j in (0, t.input_size) {
            sum += input[j] * t.weights1[i * t.input_size + j];
        });
        t.hidden[i] = act(sum + t.biases[0]);
    });

    // Calculate output layer neuron values.
    const_for!(for i in (0, t.output.len()) {
        let mut sum = 0.0;
        const_for!(for j in (0, t.hidden.len()) {
            sum += t.hidden[j] * t.weights2[i * t.hidden.len() + j];
        });
        t.output[i] = act(sum + t.biases[1]);
    })
}

// Returns an output prediction given an input.
const fn xtpredict<'t>(t: &'t mut Tinn, input: &[f64]) -> &'static [f64] {
    fprop(t, input);
    unsafe {
        let new = allocate_array(t.output.len());
        core::ptr::copy(t.output.as_ptr(), new.as_mut_ptr(), t.output.len());
        new
    }
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
const fn xttrain(t: &mut Tinn, input: &[f64], target: &[f64], rate: f64) -> f64 {
    fprop(t, input);
    bprop(t, input, target, rate);
    toterr(target, t.output)
}

const fn shuffle<T>(data: &mut [T]) {
    let mut seed = 130753105;
    const_for!(for a in (0, data.len()) {
        seed = frand(seed).0;

        let b = (seed as usize) % data.len();
        data.swap(a, b);
    });
}

fn main() {
    let (target, pred, rate, error) = unsafe {
        //let data = include!("../semeion.data");

        #[allow(non_upper_case_globals)]
        static mut data: [([f64; 256], [f64; 10]); 1593] = include!("../semeion.data");

        let hidden_size = 28;
        let anneal = 0.99;
        let iterations = 256;

        let mut rate = 1.0;
        let mut tinn = Tinn::build(data[0].0.len(), hidden_size, data[0].1.len());
        let mut last_error = 0.0;

        const_for!(for i in (0, iterations) {
            shuffle(&mut data);

            let mut error = 0.0;
            const_for!(for j in (0, data.len()) {
                let (input, target) = &data[j];
                error += xttrain(&mut tinn, input, target, rate);
            });

            rate *= anneal;
            last_error = error;
        });

        (
            &data[0].1,
            xtpredict(&mut tinn, &data[0].0),
            rate,
            last_error / data.len() as f64,
        )
    };

    println!("target: {:?}", target);
    println!("pred: {:?}", pred);
    println!("err: {:?}", error);
    println!("rate: {:?}", rate);
}
