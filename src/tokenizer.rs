use std::collections::{BTreeSet, HashMap};

pub struct Tokenizer {
    uchars: Vec<char>,
    char_to_id: HashMap<char, usize>,
    pub bos: usize,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn new(docs: &[String]) -> Self {
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
    pub fn encode(&self, doc: &str) -> Vec<usize> {
        let mut tokens = vec![self.bos];
        tokens.extend(doc.chars().map(|c| self.char_to_id[&c]));
        tokens.push(self.bos);
        tokens
    }

    /// Decode a single token id back to a char.
    pub fn decode(&self, token_id: usize) -> char {
        self.uchars[token_id]
    }
}
