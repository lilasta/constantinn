#![feature(const_for)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_heap)]
#![feature(const_mut_refs)]
#![feature(const_slice_from_raw_parts_mut)]
#![feature(core_intrinsics)]
#![feature(decl_macro)]
#![feature(inline_const)]
#![feature(const_eval_limit)]
#![const_eval_limit = "0"]

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
    //1.0 / (1.0 + (-x).exp())
    x / (1.0 + abs(x))
}

// Returns partial derivative of activation function.
const fn pdact(a: f64) -> f64 {
    a * (1.0 - a)
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

    const_for!(for i in (0, t.nw) {
        let (ns, r) = frand(seed);
        seed = ns;
        t.w[i] = r - 0.5;
    });

    const_for!(for i in (0, t.nb) {
        let (ns, r) = frand(seed);
        seed = ns;
        t.b[i] = r - 0.5;
    });
}

struct Tinn {
    // All the weights.
    w: &'static mut [f64],
    // Hidden to output layer weights.
    x: &'static mut [f64],
    // Biases.
    b: &'static mut [f64],
    // Hidden layer.
    h: &'static mut [f64],
    // Output layer.
    o: &'static mut [f64],
    // Number of biases - always two - Tinn only supports a single hidden layer.
    nb: usize,
    // Number of weights.
    nw: usize,
    // Number of inputs.
    nips: usize,
    // Number of hidden neurons.
    nhid: usize,
    // Number of outputs.
    nops: usize,
}

const unsafe fn allocate_array(size: usize) -> &'static mut [f64] {
    core::slice::from_raw_parts_mut(
        core::intrinsics::const_allocate(
            core::mem::size_of::<f64>() * size,
            core::mem::align_of::<f64>(),
        )
        .cast::<f64>(),
        size,
    )
}

impl Tinn {
    // Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
    pub const fn build(nips: usize, nhid: usize, nops: usize) -> Self {
        let nb = 2;
        let nw = nhid * (nips + nops);
        let w = unsafe { allocate_array(nw) };
        let x = unsafe {
            let skip = nhid * nips;
            core::slice::from_raw_parts_mut(w.as_mut_ptr().add(skip), nw - skip)
        };
        let mut this = Self {
            w,
            x,
            b: unsafe { allocate_array(nb) },
            h: unsafe { allocate_array(nhid) },
            o: unsafe { allocate_array(nops) },
            nb,
            nw,
            nips,
            nhid,
            nops,
        };
        wbrand(&mut this);
        this
    }
}

// Performs back propagation.
const fn bprop(t: &mut Tinn, input: &[f64], target: &[f64], rate: f64) {
    const_for!(for i in (0, t.nhid) {
        let mut sum = 0.0;
        const_for!(for j in (0, t.nops) {
            let a = pderr(t.o[j], target[j]);
            let b = pdact(t.o[j]);
            sum += a * b * t.x[j * t.nhid + i];
            // Correct weights in hidden to output layer.
            t.x[j * t.nhid + i] -= rate * a * b * t.h[i];
        });

        // Correct weights in input to hidden layer.
        const_for!(for j in (0, t.nips) {
            t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * input[j];
        });
    })
}

// Performs forward propagation.
const fn fprop(t: &mut Tinn, input: &[f64]) {
    // Calculate hidden layer neuron values.
    const_for!(for i in (0, t.nhid) {
        let mut sum = 0.0;
        const_for!(for j in (0, t.nips) {
            sum += input[j] * t.w[i * t.nips + j];
        });
        t.h[i] = act(sum + t.b[0]);
    });

    // Calculate output layer neuron values.
    const_for!(for i in (0, t.nops) {
        let mut sum = 0.0;
        const_for!(for j in (0, t.nhid) {
            sum += t.h[j] * t.x[i * t.nhid + j];
        });
        t.o[i] = act(sum + t.b[1]);
    })
}

// Returns an output prediction given an input.
const fn xtpredict<'t>(t: &'t mut Tinn, input: &[f64]) -> &'static [f64] {
    fprop(t, input);
    unsafe {
        let new = allocate_array(t.o.len());
        core::ptr::copy(t.o.as_ptr(), new.as_mut_ptr(), t.o.len());
        new
    }
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
const fn xttrain(t: &mut Tinn, input: &[f64], target: &[f64], rate: f64) -> f64 {
    fprop(t, input);
    bprop(t, input, target, rate);
    toterr(target, t.o)
}

fn main() {
    let (pred, rate, error) = const {
        let nips = 256;
        let nops = 10;
        let nhid = 28;
        let anneal = 0.99;
        let iterations = 128;

        let data = include!("../semeion.data");
        //let data = [0];
        let mut rate = 1.0;
        let mut tinn = Tinn::build(nips, nhid, nops);
        let mut last_error = 0.0;
        const_for!(for i in (0, iterations) {
            let mut error = 0.0;
            const_for!(for j in (0, data.len()) {
                let input = &data[j].0;
                let target = &data[j].1;
                error += xttrain(&mut tinn, input, target, rate);
            });
            rate *= anneal;
            last_error = error;
        });

        (xtpredict(&mut tinn, &data[0].0), rate, last_error)
    };

    println!("{:?}, {:?}, {:?}", pred, rate, error);
}
