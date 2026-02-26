use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::ops::{rms_norm, softmax};

pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl Config {
    pub fn new(vocab_size: usize) -> Self {
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

pub struct Layer {
    pub attn_wq: Var,
    pub attn_wk: Var,
    pub attn_wv: Var,
    pub attn_wo: Var,
    pub mlp_fc1: Var,
    pub mlp_fc2: Var,
}

pub struct Model {
    pub wte: Var,     // token embeddings    [vocab_size, n_embd]
    pub wpe: Var,     // position embeddings [block_size, n_embd]
    pub lm_head: Var, // output projection   [vocab_size, n_embd]
    pub layers: Vec<Layer>,
}

pub struct KvCache {
    pub keys: Vec<Vec<Tensor>>,   // [n_layer][seq_len] of [n_embd]
    pub values: Vec<Vec<Tensor>>, // [n_layer][seq_len] of [n_embd]
}

impl KvCache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            keys: vec![vec![]; n_layer],
            values: vec![vec![]; n_layer],
        }
    }
}

// --- helpers ---

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

// --- model ---

impl Model {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
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
    pub fn params(&self) -> Vec<Var> {
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

    pub fn num_params(&self) -> usize {
        self.params().iter().map(|v| v.as_tensor().elem_count()).sum()
    }

    /// Forward one token, updating the KV cache. Returns logits [vocab_size].
    pub fn forward(
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
