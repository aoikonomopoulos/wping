use std::cmp;
use std::fmt::Display;

pub struct Column {
    pub name : String,
    // The maximum width of the column (max(header, value))
    pub fmt_max_width : usize,
    // Minumum whitespace between this column and adjacent ones.
    // E.g. if the column name contains a space it might be necessary
    // to increase the breathing whitespace, so that the grouping of
    // words into column names is apparent to the viewer.
    pub breathing_whitespace : u8,
}

impl Column {
    pub fn new(name : &str, fmt_max_width : usize, breathing_whitespace : u8) -> Column {
        Column {
            name : name.to_string(),
            fmt_max_width : fmt_max_width,
            breathing_whitespace : breathing_whitespace,
        }
    }
}

pub struct Columnar {
    columns : Vec<Column>,
}

// XXX: switch to String.repeat when it stabilizes
fn spaces(n : u8) -> String {
    let mut acc = String::new();
    for _ in 0..n {
        acc.push_str(" ")
    }
    acc
}

fn breathing_whitespace(prev : Option<&Column>, curr : &Column) -> String {
    let bw = match prev {
        None =>
            0,
        Some (prev) =>
            cmp::max(prev.breathing_whitespace, curr.breathing_whitespace)
    };
    spaces(bw)
}

impl<'a> Columnar {
    pub fn new() -> Columnar {
        Columnar {
            columns : Vec::new(),
        }
    }
    pub fn push_col(mut self, c : Column) -> Columnar {
        self.columns.push(c);
        self
    }
    pub fn header(&'a self) -> String {
        (&self.columns).into_iter().
            fold(("".to_string(), None),
                 |(mut acc, prev_col) : (String, Option<&'a Column>), col : &'a Column| {
                     acc.push_str(&breathing_whitespace(prev_col, col));
                     acc.push_str(&format!("{0:^1$}", col.name, col.fmt_max_width));
                     (acc, Some (col))
            }).0
    }
    pub fn format(&'a self, values : Vec<Option<&Display>>) -> String {
        let na = "-".to_string();
        self.columns.iter().zip(values).
            fold((String::new(), None),
                 |(mut acc, prev_col) : (String, Option<&'a Column>),
                 (col, v) : (&'a Column, Option<&Display>)| {
                     acc.push_str(&breathing_whitespace(prev_col, col));
                     let v = v.unwrap_or(&na);
                     acc.push_str(&format!("{0:^1$}", v, col.fmt_max_width));
                     (acc, Some (col))
                 }).0
    }
}
