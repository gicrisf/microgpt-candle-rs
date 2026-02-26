mod model;
mod tokenizer;

use candle_core::{Device, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use model::{Config, KvCache, Model};
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use std::fs;
use std::io::{self, Write};
use tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
