
type Var = u64;

#[derive(Debug,Clone)]
pub struct Ewma {
    pub smoothed : Var,
    pub variation : Var,
}

const ALPHA_NOM : Var = 1 as Var;
const ALPHA_DENOM : Var = 8 as Var;
const BETA_NOM : Var = 1 as Var;
const BETA_DENOM : Var = 4 as Var;

// Simple implementation of an exponential weighted moving average algorithm.
// Heavy use of 'as' to convert between integer types, so that we can freely switch
// what Var aliases to. Might be better to make this generic over some numeric T.
impl Ewma {
    pub fn new(sample : u64) -> Ewma {
        Ewma {
            smoothed : (sample as Var),
            variation : (sample as Var) / (2 as Var),
        }
    }
    // This tries to implement RTT estimation according to the letter of
    // RFC6298. Any deviation is unintentional.
    pub fn add_sample(&mut self, sample_ : u64) {
        let sample = sample_ as Var;
        let distance = if self.smoothed > sample {
            self.smoothed - sample
        } else {
            sample - self.smoothed
        };
        self.variation = (self.variation - (BETA_NOM * self.variation) / BETA_DENOM) +
            (BETA_NOM * distance) / BETA_DENOM;
        self.smoothed = (self.smoothed - (ALPHA_NOM * self.smoothed) / ALPHA_DENOM) +
            (ALPHA_NOM * sample) / ALPHA_DENOM;
    }
}
