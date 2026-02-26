use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::ops::{log_softmax, rms_norm, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{self, Write};

// --- Tokenizer ---

struct Tokenizer {
    uchars: Vec<char>,
    char_to_id: HashMap<char, usize>,
    bos: usize,
    vocab_size: usize,
}

impl Tokenizer {
    fn new(docs: &[String]) -> Self {
        let uchars: Vec<char> = docs
            .iter()
            .flat_map(|d| d.chars())
            .collect::<BTreeSet<char>>()
            .into_iter()
            .collect();
        let char_to_id: HashMap<char, usize> =
            uchars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let bos = uchars.len();
        let vocab_size = uchars.len() + 1;
        Self { uchars, char_to_id, bos, vocab_size }
    }

    /// Encode a document into token ids, surrounded by BOS on both sides.
    fn encode(&self, doc: &str) -> Vec<usize> {
        let mut tokens = vec![self.bos];
        tokens.extend(doc.chars().map(|c| self.char_to_id[&c]));
        tokens.push(self.bos);
        tokens
    }

    /// Decode a single token id back to a char.
    fn decode(&self, token_id: usize) -> char {
        self.uchars[token_id]
    }
}

// --- Model ---

struct Config {
    n_layer: usize,
    n_embd: usize,
    block_size: usize,
    n_head: usize,
    head_dim: usize,
    vocab_size: usize,
}

impl Config {
    fn new(vocab_size: usize) -> Self {
        let n_embd = 16;
        let n_head = 4;
        Self {
            n_layer: 1,
            n_embd,
            block_size: 16,
            n_head,
            head_dim: n_embd / n_head,
            vocab_size,
        }
    }
}

struct Layer {
    attn_wq: Var,
    attn_wk: Var,
    attn_wv: Var,
    attn_wo: Var,
    mlp_fc1: Var,
    mlp_fc2: Var,
}

struct Model {
    wte: Var,     // token embeddings    [vocab_size, n_embd]
    wpe: Var,     // position embeddings [block_size, n_embd]
    lm_head: Var, // output projection   [vocab_size, n_embd]
    layers: Vec<Layer>,
}

struct KvCache {
    keys: Vec<Vec<Tensor>>,   // [n_layer][seq_len] of [n_embd]
    values: Vec<Vec<Tensor>>, // [n_layer][seq_len] of [n_embd]
}

impl KvCache {
    fn new(n_layer: usize) -> Self {
        Self {
            keys: vec![vec![]; n_layer],
            values: vec![vec![]; n_layer],
        }
    }
}

fn randn_var(nout: usize, nin: usize, device: &Device) -> Result<Var> {
    Var::from_tensor(&Tensor::randn(0f32, 0.08f32, (nout, nin), device)?)
}

fn rmsnorm(x: &Tensor) -> Result<Tensor> {
    let alpha = Tensor::ones(x.dim(0)?, DType::F32, x.device())?;
    rms_norm(x, &alpha, 1e-5f32)
}

fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    w.matmul(&x.unsqueeze(1)?)?.squeeze(1)
}

impl Model {
    fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let mat = |nout, nin| randn_var(nout, nin, device);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(Layer {
                attn_wq: mat(cfg.n_embd, cfg.n_embd)?,
                attn_wk: mat(cfg.n_embd, cfg.n_embd)?,
                attn_wv: mat(cfg.n_embd, cfg.n_embd)?,
                attn_wo: mat(cfg.n_embd, cfg.n_embd)?,
                mlp_fc1: mat(4 * cfg.n_embd, cfg.n_embd)?,
                mlp_fc2: mat(cfg.n_embd, 4 * cfg.n_embd)?,
            });
        }

        Ok(Self {
            wte: mat(cfg.vocab_size, cfg.n_embd)?,
            wpe: mat(cfg.block_size, cfg.n_embd)?,
            lm_head: mat(cfg.vocab_size, cfg.n_embd)?,
            layers,
        })
    }

    /// All trainable variables, in the same order as Python's `params`.
    fn params(&self) -> Vec<Var> {
        let mut vars = vec![self.wte.clone(), self.wpe.clone(), self.lm_head.clone()];
        for l in &self.layers {
            vars.extend([
                l.attn_wq.clone(),
                l.attn_wk.clone(),
                l.attn_wv.clone(),
                l.attn_wo.clone(),
                l.mlp_fc1.clone(),
                l.mlp_fc2.clone(),
            ]);
        }
        vars
    }

    fn num_params(&self) -> usize {
        self.params().iter().map(|v| v.as_tensor().elem_count()).sum()
    }

    /// Forward one token, updating the KV cache. Returns logits [vocab_size].
    fn forward(
        &self,
        cfg: &Config,
        token_id: usize,
        pos_id: usize,
        kv: &mut KvCache,
    ) -> Result<Tensor> {
        let tok_emb = self.wte.as_tensor().get(token_id)?; // [n_embd]
        let pos_emb = self.wpe.as_tensor().get(pos_id)?;   // [n_embd]
        let mut x = tok_emb.add(&pos_emb)?;
        x = rmsnorm(&x)?;

        for li in 0..cfg.n_layer {
            let layer = &self.layers[li];

            // --- Attention ---
            let x_residual = x.clone();
            x = rmsnorm(&x)?;
            let q = linear(&x, layer.attn_wq.as_tensor())?;
            let k = linear(&x, layer.attn_wk.as_tensor())?;
            let v = linear(&x, layer.attn_wv.as_tensor())?;

            kv.keys[li].push(k);
            kv.values[li].push(v);

            let k_all = Tensor::stack(&kv.keys[li], 0)?; // [seq_len, n_embd]
            let v_all = Tensor::stack(&kv.values[li], 0)?;

            let scale = (cfg.head_dim as f64).sqrt();
            let mut head_outs = Vec::with_capacity(cfg.n_head);
            for h in 0..cfg.n_head {
                let hs = h * cfg.head_dim;
                let q_h = q.narrow(0, hs, cfg.head_dim)?;     // [head_dim]
                let k_h = k_all.narrow(1, hs, cfg.head_dim)?; // [seq_len, head_dim]
                let v_h = v_all.narrow(1, hs, cfg.head_dim)?; // [seq_len, head_dim]

                let attn_logits = k_h
                    .matmul(&q_h.unsqueeze(1)?)? // [seq_len, 1]
                    .squeeze(1)?                  // [seq_len]
                    .affine(1.0 / scale, 0.0)?;

                let attn_weights = softmax(&attn_logits, 0)?; // [seq_len]

                let head_out = v_h
                    .t()?                                  // [head_dim, seq_len]
                    .matmul(&attn_weights.unsqueeze(1)?)? // [head_dim, 1]
                    .squeeze(1)?;                          // [head_dim]

                head_outs.push(head_out);
            }

            let x_attn = Tensor::cat(&head_outs, 0)?; // [n_embd]
            x = linear(&x_attn, layer.attn_wo.as_tensor())?;
            x = x.add(&x_residual)?;

            // --- MLP ---
            let x_residual = x.clone();
            x = rmsnorm(&x)?;
            x = linear(&x, layer.mlp_fc1.as_tensor())?; // [4*n_embd]
            x = x.relu()?;
            x = linear(&x, layer.mlp_fc2.as_tensor())?; // [n_embd]
            x = x.add(&x_residual)?;
        }

        linear(&x, self.lm_head.as_tensor()) // [vocab_size]
    }
}

// --- Main ---

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string("input.txt")?;
    let mut docs: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    docs.shuffle(&mut rand::thread_rng());
    println!("num docs: {}", docs.len());

    let tok = Tokenizer::new(&docs);
    println!("vocab size: {}", tok.vocab_size);

    let device = Device::Cpu;
    let cfg = Config::new(tok.vocab_size);
    let model = Model::new(&cfg, &device)?;
    println!("num params: {}", model.num_params());

    let num_steps = 1000;
    let learning_rate = 0.01f64;
    let mut optimizer = AdamW::new(
        model.params(),
        ParamsAdamW { lr: learning_rate, beta1: 0.85, beta2: 0.99, eps: 1e-8, weight_decay: 0.0 },
    )?;

    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let tokens = tok.encode(doc);
        let n = cfg.block_size.min(tokens.len() - 1);

        // Forward: one token at a time, accumulating cross-entropy losses
        let mut kv = KvCache::new(cfg.n_layer);
        let mut losses: Vec<Tensor> = Vec::with_capacity(n);
        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = model.forward(&cfg, token_id, pos_id, &mut kv)?;
            // log_softmax is numerically more stable than softmax + log
            let log_probs = log_softmax(&logits, 0)?;
            let loss_t = log_probs.narrow(0, target_id, 1)?.neg()?; // [1]
            losses.push(loss_t);
        }

        // Average loss over the sequence, then backward + Adam step
        let loss = Tensor::cat(&losses, 0)?.mean(0)?;
        let lr = learning_rate * (1.0 - step as f64 / num_steps as f64);
        optimizer.set_learning_rate(lr);
        optimizer.backward_step(&loss)?;

        print!("\rstep {:4} / {:4} | loss {:.4}", step + 1, num_steps, loss.to_scalar::<f32>()?);
        io::stdout().flush()?;
    }
    println!();

    // --- Inference ---
    let temperature = 0.5f64;
    let mut rng = rand::thread_rng();
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut kv = KvCache::new(cfg.n_layer);
        let mut token_id = tok.bos;
        let mut sample = String::new();

        for pos_id in 0..cfg.block_size {
            let logits = model.forward(&cfg, token_id, pos_id, &mut kv)?;
            let probs = softmax(&logits.affine(1.0 / temperature, 0.0)?, 0)?;
            let weights = probs.to_vec1::<f32>()?;
            token_id = WeightedIndex::new(&weights)?.sample(&mut rng);
            if token_id == tok.bos {
                break;
            }
            sample.push(tok.decode(token_id));
        }

        println!("sample {:2}: {}", sample_idx + 1, sample);
    }

    Ok(())
}
