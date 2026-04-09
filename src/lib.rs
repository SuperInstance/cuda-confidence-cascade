//! cuda-confidence-cascade — GPU confidence propagation

#[derive(Debug, Clone, Copy)]
pub struct ConfVal { pub value: f64, pub confidence: f64 }

impl ConfVal {
    pub fn new(v: f64, c: f64) -> Self { ConfVal { value: v, confidence: c.clamp(0.0, 1.0) } }
    pub fn certain(v: f64) -> Self { ConfVal::new(v, 1.0) }
    pub fn unknown() -> Self { ConfVal::new(0.0, 0.0) }
    pub fn combine(&self, other: &ConfVal, op: impl Fn(f64, f64) -> f64) -> ConfVal {
        ConfVal::new(op(self.value, other.value), self.confidence * other.confidence)
    }
    pub fn or_combine(&self, other: &ConfVal) -> ConfVal {
        ConfVal::new(if self.confidence >= other.confidence { self.value } else { other.value },
            1.0 - (1.0 - self.confidence) * (1.0 - other.confidence))
    }
}

pub struct ConfidenceCascade { pub values: Vec<ConfVal>, pub operations: Vec<String> }

impl ConfidenceCascade {
    pub fn new() -> Self { ConfidenceCascade { values: Vec::new(), operations: Vec::new() } }
    pub fn push(&mut self, val: ConfVal) { self.values.push(val); }
    pub fn arithmetic(&mut self, op: &str) -> ConfVal {
        if self.values.len() < 2 { return ConfVal::unknown(); }
        let b = self.values.pop().unwrap(); let a = self.values.pop().unwrap();
        let result = match op {
            "+" => a.combine(&b, |x,y| x+y), "-" => a.combine(&b, |x,y| x-y),
            "*" => a.combine(&b, |x,y| x*y), "/" => {
                if b.value.abs() > 1e-10 { a.combine(&b, |x,y| x/y) } else { ConfVal::unknown() }
            } _ => ConfVal::unknown(),
        };
        self.operations.push(format!("{} {} {} = {:.3} (c={:.3})", a.value, op, b.value, result.value, result.confidence));
        self.values.push(result); result
    }
    pub fn bayesian_update(prior: &ConfVal, evidence: &ConfVal) -> ConfVal {
        let post_conf = 1.0 / (1.0 / prior.confidence.max(0.001) + 1.0 / evidence.confidence.max(0.001));
        let post_val = prior.value * evidence.confidence + evidence.value * prior.confidence;
        ConfVal::new(post_val / (prior.confidence + evidence.confidence).max(0.001), post_conf)
    }
}

#[cfg(test)]
mod tests { use super::*;
    #[test] fn test_propagation() { let c = ConfVal::new(5.0, 0.8).combine(&ConfVal::new(3.0, 0.9), |x,y| x+y); assert!((c.value-8.0).abs()<0.01); assert!((c.confidence-0.72).abs()<0.01); }
    #[test] fn test_bayesian() { let p = ConfVal::new(0.5, 0.3); let e = ConfVal::new(0.8, 0.7);
        let post = ConfidenceCascade::bayesian_update(&p, &e); assert!(post.confidence > p.confidence); }
}
