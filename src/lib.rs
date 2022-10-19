//! AIGER (And-Inverter graph) file format parser
#![deny(missing_docs)]

use std::io::{self, BufRead};
use std::str::FromStr;

/// A literal value from an AIGER file, encoding both a variable index and "sign
/// bit" which determines if the variable is negated/inverted.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash)]
#[repr(transparent)]
pub struct Literal(pub usize);

impl Literal {
    /// Builds a literal out of a variable index and sign bit.
    pub fn from_variable(variable: usize, is_inverted: bool) -> Literal {
        Literal(variable * 2 + if is_inverted { 1 } else { 0 })
    }

    /// Returns the variable the literal refers to.
    pub fn variable(&self) -> usize {
        self.0 / 2
    }

    /// Returns true if the literal inverts the variable, or false if it does
    /// not.
    pub fn is_inverted(&self) -> bool {
        (self.0 & 1) == 1
    }
}

/// The data contained in the header of an AIGER file.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub struct Header {
    /// The maximum variable index.
    pub m: usize,
    /// The number of inputs.
    pub i: usize,
    /// The number of latches.
    pub l: usize,
    /// The number of outputs.
    pub o: usize,
    /// The number of AND gates.
    pub a: usize,
    /// The bad state
    pub b: usize,
}

impl FromStr for Header {
    type Err = AigerError;

    // In the interest of matching both the header structure and the naming
    // convention the format itself uses, we'll use the M I L O A names.
    #[allow(clippy::many_single_char_names)]
    fn from_str(header_line: &str) -> Result<Self, Self::Err> {
        let mut components = header_line.split(' ');
        let magic = components.next().ok_or(AigerError::InvalidHeader)?;

        const HEADER_MAGIC: &str = "aag";
        if magic != HEADER_MAGIC {
            return Err(AigerError::InvalidHeader);
        }

        let mut components =
            components.map(|s| usize::from_str_radix(s, 10).map_err(|_| AigerError::InvalidHeader));

        // The remaining components of the header are all integers
        let mut get_component = || components.next().ok_or(AigerError::InvalidHeader)?;
        let m = get_component()?;
        let i = get_component()?;
        let l = get_component()?;
        let o = get_component()?;
        let a = get_component()?;
        let b = match get_component() {
            Ok(b) => b,
            Err(_) => 0,
        };

        if components.next() != None {
            // We have extra components after what should've been the last
            // component
            Err(AigerError::InvalidHeader)
        } else {
            Ok(Header { m, i, l, o, a, b })
        }
    }
}

/// The type specifier for a symbol table entry.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub enum Symbol {
    /// The symbol names an input.
    Input,
    /// The symbol names a latch.
    Latch,
    /// The symbol names an output.
    Output,
}

/// A record from an AIGER file.
#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub enum Aiger {
    /// A literal marked as an input.
    Input(Literal),
    /// A latch.
    Latch {
        /// The literal which receives the latch's current state.
        output: Literal,
        /// The literal which determines the latch's next state.
        input: Literal,
        /// The init value of latch
        init: bool,
    },
    /// A literal marked as an output.
    Output(Literal),
    /// A literal marked as a bad state.
    BadState(Literal),
    /// An AND gate.
    AndGate {
        /// The literal which receives the result of the AND operation.
        output: Literal,
        /// The literals which are inputs to the AND gate.
        inputs: [Literal; 2],
    },
    /// An entry from the symbol table.
    Symbol {
        /// The type specifier for the symbol.
        type_spec: Symbol,
        /// The position in the file of the input/latch/output that this symbol
        /// labels.
        ///
        /// Though latches are listed in an AIGER file after inputs, and outputs
        /// after both inputs and latches, this position value is the index of
        /// the record within the records of the same type.
        ///
        /// That is:
        /// ```text
        /// aag 8 2 2 2 0
        /// // inputs
        /// 2     // position = 0
        /// 4     // position = 1
        /// // latches
        /// 6 8   // position = 0
        /// 10 12 // position = 1
        /// // and gates (cannot be assigned symbols)
        /// ...
        /// // outputs
        /// 14    // position = 0
        /// 16    // position = 1
        /// ```
        position: usize,
        /// The actual symbol.
        symbol: String,
    },
}

impl Aiger {
    /// Ensures the literals within the record are valid, returning the record
    /// if so or an error if one was detected.
    fn validate(self, header_m: usize) -> Result<Aiger, AigerError> {
        match self {
            Aiger::Input(l) if l.is_inverted() => return Err(AigerError::InvalidInverted),
            Aiger::Latch { output, .. } if output.is_inverted() => {
                return Err(AigerError::InvalidInverted)
            }
            Aiger::AndGate { output, .. } if output.is_inverted() => {
                return Err(AigerError::InvalidInverted)
            }
            _ => {}
        }

        let literals = match self {
            Aiger::Input(l) => vec![l],
            Aiger::Latch {
                output,
                input,
                init,
            } => vec![output, input, Literal(init.into())],
            Aiger::Output(l) => vec![l],
            Aiger::BadState(l) => vec![l],
            Aiger::AndGate {
                output,
                inputs: [input0, input1],
            } => vec![output, input0, input1],
            Aiger::Symbol { .. } => return Ok(self),
        };

        for literal in literals {
            if literal.variable() > header_m {
                return Err(AigerError::LiteralOutOfRange);
            }
        }

        Ok(self)
    }

    fn parse_input(literals: &[Literal]) -> Result<Aiger, AigerError> {
        match literals {
            [input] => Ok(Aiger::Input(*input)),
            _ => Err(AigerError::InvalidLiteralCount),
        }
    }

    fn parse_latch(literals: &[Literal]) -> Result<Aiger, AigerError> {
        match literals {
            [output, input] => Ok(Aiger::Latch {
                output: *output,
                input: *input,
                init: false,
            }),
            [output, input, init] => Ok(Aiger::Latch {
                output: *output,
                input: *input,
                init: init.0 != 0,
            }),
            _ => Err(AigerError::InvalidLiteralCount),
        }
    }

    fn parse_output(literals: &[Literal]) -> Result<Aiger, AigerError> {
        match literals {
            [input] => Ok(Aiger::Output(*input)),
            _ => Err(AigerError::InvalidLiteralCount),
        }
    }

    fn parse_badstate(literals: &[Literal]) -> Result<Aiger, AigerError> {
        match literals {
            [input] => Ok(Aiger::BadState(*input)),
            _ => Err(AigerError::InvalidLiteralCount),
        }
    }

    fn parse_and_gate(literals: &[Literal]) -> Result<Aiger, AigerError> {
        match literals {
            [output, input1, input2] => Ok(Aiger::AndGate {
                output: *output,
                inputs: [*input1, *input2],
            }),
            _ => Err(AigerError::InvalidLiteralCount),
        }
    }

    fn parse_symbol(line: &str) -> Result<Aiger, AigerError> {
        let (type_spec, rest) = line.split_at(1);
        let type_spec = match type_spec {
            "i" => Symbol::Input,
            "l" => Symbol::Latch,
            "o" => Symbol::Output,
            _ => return Err(AigerError::InvalidSymbol),
        };

        let space_position = rest.find(' ').ok_or(AigerError::InvalidSymbol)?;
        let (position, rest) = rest.split_at(space_position);
        let position =
            usize::from_str_radix(position, 10).map_err(|_| AigerError::InvalidSymbol)?;

        let (_, symbol) = rest.split_at(1);

        if symbol.is_empty() {
            return Err(AigerError::InvalidSymbol);
        }

        Ok(Aiger::Symbol {
            type_spec,
            position,
            symbol: symbol.to_string(),
        })
    }
}

/// An error which occurs while parsing an AIGER file.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub enum AigerError {
    /// No AIGER header could be found, or the header which was found could not
    /// be parsed.
    InvalidHeader,
    /// A literal which was not a positive integer was encountered.
    InvalidLiteral,
    /// A literal with a variable greater than the maximum declared in the AIGER
    /// header was encountered.
    LiteralOutOfRange,
    /// Too many or too few literals were encountered for the expected type of
    /// record.
    InvalidLiteralCount,
    /// An inverted literal was encountered where an inverted literal is not
    /// allowed.
    InvalidInverted,
    /// An invalid symbol table entry was encountered.
    InvalidSymbol,
    /// An IO error occurred while reading.
    IoError,
}

impl From<io::Error> for AigerError {
    fn from(_error: io::Error) -> Self {
        AigerError::IoError
    }
}

/// A wrapper around a type implementing `io::Read` which reads an AIGER header
/// and AIGER records.
pub struct Reader<T: io::Read> {
    /// The AIGER header which was parsed during reader construction.
    header: Header,
    lines: io::Lines<io::BufReader<T>>,
}

impl<T: io::Read> std::fmt::Debug for Reader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("header", &self.header)
            .finish()
    }
}

impl<T: io::Read> Reader<T> {
    /// Creates a new AIGER reader which reads from the provided reader.
    ///
    /// # Example
    /// ```
    /// use aiger::Reader;
    /// let readable = "aag 3 2 0 1 0\n2\n4\n6\n6 2 4\n".as_bytes();
    /// let reader = Reader::from_reader(readable).unwrap();
    ///
    /// println!("{:?}", reader.header());
    ///
    /// for record in reader.records() {
    ///     println!("{:?}", record);
    /// }
    /// ```
    pub fn from_reader(reader: T) -> Result<Reader<T>, AigerError> {
        let reader = io::BufReader::new(reader);
        let mut lines = reader.lines();

        let header_line = lines.next().ok_or(AigerError::InvalidHeader)??;
        let header = header_line.parse::<Header>()?;

        Ok(Reader { header, lines })
    }

    /// Returns an iterator over the records in the AIGER file, consuming the
    /// reader.
    pub fn records(self) -> RecordsIter<T> {
        RecordsIter::new(self.lines, self.header)
    }

    /// Returns the AIGER header.
    pub fn header(&self) -> Header {
        self.header
    }
}

/// An iterator over the records of an AIGER file.
pub struct RecordsIter<T: io::Read> {
    /// The header of the AIGER file.
    header: Header,
    /// An iterator over the lines of the AIGER file.
    lines: io::Lines<io::BufReader<T>>,
    /// Number of inputs which are yet to be parsed.
    remaining_inputs: usize,
    /// Number of outputs which are yet to be parsed.
    remaining_latches: usize,
    /// Number of outputs which are yet to be parsed.
    remaining_outputs: usize,
    /// Number of AND gates which are yet to be parsed.
    remaining_and_gates: usize,
    /// Number of AND gates which are yet to be parsed.
    remaining_bad_states: usize,
    /// True if we have reached a comment in the file.
    comment_reached: bool,
}

impl<T: io::Read> RecordsIter<T> {
    fn new(lines: io::Lines<io::BufReader<T>>, header: Header) -> RecordsIter<T> {
        RecordsIter {
            lines,
            header,
            remaining_inputs: header.i,
            remaining_latches: header.l,
            remaining_outputs: header.o,
            remaining_and_gates: header.a,
            remaining_bad_states: header.b,
            comment_reached: false,
        }
    }

    fn read_record(&mut self, line: &str) -> Result<Aiger, AigerError> {
        let get_literals = || -> Result<Vec<Literal>, AigerError> {
            Ok(line
                .split(' ')
                .map(|s| usize::from_str_radix(s, 10).map(Literal))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| AigerError::InvalidLiteral)?)
        };

        if self.remaining_inputs > 0 {
            self.remaining_inputs -= 1;
            Aiger::parse_input(&get_literals()?)
        } else if self.remaining_latches > 0 {
            self.remaining_latches -= 1;
            Aiger::parse_latch(&get_literals()?)
        } else if self.remaining_outputs > 0 {
            self.remaining_outputs -= 1;
            Aiger::parse_output(&get_literals()?)
        } else if self.remaining_bad_states > 0 {
            self.remaining_bad_states -= 1;
            Aiger::parse_badstate(&get_literals()?)
        } else if self.remaining_and_gates > 0 {
            self.remaining_and_gates -= 1;
            Aiger::parse_and_gate(&get_literals()?)
        } else {
            Aiger::parse_symbol(line)
        }
    }
}

impl<T: io::Read> Iterator for RecordsIter<T> {
    type Item = Result<Aiger, AigerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.comment_reached {
            return None;
        }

        let line = match self.lines.next() {
            Some(line) => line,
            None => return None,
        };

        let line = match line {
            Ok(line) => line,
            Err(e) => return Some(Err(e.into())),
        };

        if let Some('c') = line.chars().next() {
            self.comment_reached = true;
            return None;
        }

        Some(
            self.read_record(&line)
                .and_then(|record| record.validate(self.header.m)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reader(s: &'static str) -> Result<Reader<&[u8]>, AigerError> {
        Reader::from_reader(s.as_bytes())
    }

    #[test]
    fn literal() {
        for (literal, variable, is_inverted) in &[
            (0, 0, false),
            (1, 0, true),
            (2, 1, false),
            (3, 1, true),
            (100, 50, false),
            (101, 50, true),
        ] {
            let literal = Literal(*literal);

            assert_eq!(literal.variable(), *variable);
            assert_eq!(literal.is_inverted(), *is_inverted);

            assert_eq!(Literal::from_variable(*variable, *is_inverted), literal);
        }
    }

    #[test]
    fn reader_no_header() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "",
            )).unwrap_err();

        assert_eq!(reader, AigerError::InvalidHeader);
    }

    #[test]
    fn reader_header_invalid_magic() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "axg 0 0 0 0 0\n",
            )).unwrap_err();

        assert_eq!(reader, AigerError::InvalidHeader);
    }

    #[test]
    fn reader_header_too_short() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 0 0 0\n",
            )).unwrap_err();

        assert_eq!(reader, AigerError::InvalidHeader);
    }

    #[test]
    fn reader_header_too_long() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 0 0 0 0 0\n",
            )).unwrap_err();

        assert_eq!(reader, AigerError::InvalidHeader);
    }

    #[test]
    fn reader_header_invalid_value() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 q 0 0 0\n",
            )).unwrap_err();

        assert_eq!(reader, AigerError::InvalidHeader);
    }

    #[test]
    fn reader_invalid_literal() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 0 0 1 0\n",
                "-5\n"
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 0,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Err(AigerError::InvalidLiteral)));
    }

    #[test]
    fn reader_invalid_literal_count_too_many() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 3 2 0 1 1\n",
                "2\n",
                "4\n",
                "6\n",
                "6 2\n", // should be 3 literals for an AND gate
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 3,
                i: 2,
                l: 0,
                o: 1,
                a: 1,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(4)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(6)))));
        assert_eq!(records.next(), Some(Err(AigerError::InvalidLiteralCount)));
    }

    #[test]
    fn reader_invalid_literal_count_too_few() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 2 1 0 1 0\n",
                "2\n",
                "4 5\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 2,
                i: 1,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Err(AigerError::InvalidLiteralCount)));
    }
    #[test]
    fn reader_invalid_inverted_input() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 1 0 0 0\n",
                "3\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 1,
                l: 0,
                o: 0,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Err(AigerError::InvalidInverted)));
    }

    #[test]
    fn reader_invalid_inverted_and() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 0 0 0 1\n",
                "3 0 1\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 0,
                l: 0,
                o: 0,
                a: 1,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Err(AigerError::InvalidInverted)));
    }

    #[test]
    fn reader_invalid_inverted_latch() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 0 1 0 0\n",
                "3 0\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 0,
                l: 1,
                o: 0,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Err(AigerError::InvalidInverted)));
    }

    #[test]
    fn reader_literal_out_of_range_input() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 2 0 0 0\n",
                "2\n",
                "4\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 2,
                l: 0,
                o: 0,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Err(AigerError::LiteralOutOfRange)));
    }

    #[test]
    fn reader_literal_out_of_range_latch() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 1 1 0 0\n",
                "2\n",
                "4 2\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 1,
                l: 1,
                o: 0,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Err(AigerError::LiteralOutOfRange)));
    }

    #[test]
    fn reader_literal_out_of_range_output() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 1 1 0 1 0\n",
                "2\n",
                "4\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 1,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Err(AigerError::LiteralOutOfRange)));
    }

    #[test]
    fn reader_literal_out_of_range_and_gate() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 2 1 0 1 1\n",
                "2\n",
                "4\n",
                "4 2 6\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 2,
                i: 1,
                l: 0,
                o: 1,
                a: 1,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(4)))));
        assert_eq!(records.next(), Some(Err(AigerError::LiteralOutOfRange)));
    }

    #[test]
    fn reader_invalid_symbol_type_spec() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 0 0 1 0\n",
                "0\n",
                "x0 zero\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 0,
                i: 0,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(0)))));
        assert_eq!(records.next(), Some(Err(AigerError::InvalidSymbol)));
    }

    #[test]
    fn reader_invalid_symbol_position() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 0 0 1 0\n",
                "0\n",
                "o-1 zero\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 0,
                i: 0,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(0)))));
        assert_eq!(records.next(), Some(Err(AigerError::InvalidSymbol)));
    }

    #[test]
    fn reader_invalid_symbol_symbol_missing() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 0 0 1 0\n",
                "0\n",
                "o0\n",
            )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 0,
                i: 0,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(0)))));
        assert_eq!(records.next(), Some(Err(AigerError::InvalidSymbol)));
    }

    #[test]
    fn reader_invalid_symbol_symbol_missing_with_space() {
        #[rustfmt::skip]
            let reader = make_reader(concat!(
                "aag 0 0 0 1 0\n",
                "0\n",
                "o0 \n",
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 0,
                i: 0,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(0)))));
        assert_eq!(records.next(), Some(Err(AigerError::InvalidSymbol)));
    }

    #[test]
    fn reader_empty_file() {
        #[rustfmt::skip]
        let reader = make_reader(concat!(
            "aag 0 0 0 0 0\n",
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 0,
                i: 0,
                l: 0,
                o: 0,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), None);
    }

    #[test]
    fn reader_single_output() {
        #[rustfmt::skip]
        let reader = make_reader(concat!(
            "aag 1 0 0 1 0\n",
            "2\n"
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 0,
                l: 0,
                o: 1,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(2)))));
        assert_eq!(records.next(), None);
    }

    #[test]
    fn reader_single_input() {
        #[rustfmt::skip]
        let reader = make_reader(concat!(
            "aag 1 1 0 0 0\n",
            "2\n",
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 1,
                i: 1,
                l: 0,
                o: 0,
                a: 0,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), None);
    }

    #[test]
    fn reader_and_gate() {
        #[rustfmt::skip]
        let reader = make_reader(concat!(
            "aag 3 2 0 1 1\n",
            "2\n",
            "4\n",
            "6\n",
            "6 2 4\n",
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 3,
                i: 2,
                l: 0,
                o: 1,
                a: 1,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(4)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(6)))));
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                inputs: [Literal(2), Literal(4)],
                output: Literal(6),
            }))
        );
        assert_eq!(records.next(), None);
    }

    #[test]
    fn reader_or_gate() {
        #[rustfmt::skip]
        let reader = make_reader(concat!(
            "aag 3 2 0 1 1\n",
            "2\n",
            "4\n",
            "7\n",
            "6 3 5\n",
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 3,
                i: 2,
                l: 0,
                o: 1,
                a: 1,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(4)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(7)))));
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                inputs: [Literal(3), Literal(5)],
                output: Literal(6),
            }))
        );
        assert_eq!(records.next(), None);
    }

    #[test]
    fn reader_half_adder() {
        #[rustfmt::skip]
        let reader = make_reader(concat!(
            "aag 7 2 0 2 3\n",
            "2\n",
            "4\n",
            "6\n",
            "12\n",
            "6 13 15\n",
            "12 2 4\n",
            "14 3 5\n",
            "i0 x\n",
            "i1 y\n",
            "o0 s\n",
            "o1 c\n",
            "c\n",
            "This is a comment.\n",
        )).unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 7,
                i: 2,
                l: 0,
                o: 2,
                a: 3,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(4)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(6)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(12)))));
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                inputs: [Literal(13), Literal(15)],
                output: Literal(6),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                inputs: [Literal(2), Literal(4)],
                output: Literal(12),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                inputs: [Literal(3), Literal(5)],
                output: Literal(14),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 0,
                type_spec: Symbol::Input,
                symbol: "x".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 1,
                type_spec: Symbol::Input,
                symbol: "y".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 0,
                type_spec: Symbol::Output,
                symbol: "s".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 1,
                type_spec: Symbol::Output,
                symbol: "c".to_string(),
            }))
        );
        assert_eq!(records.next(), None);
    }

    #[test]
    fn reader_toggle_ff_en_rst() {
        let reader = make_reader(concat!(
            "aag 7 2 1 2 4\n",
            "2\n",
            "4\n",
            "6 8\n",
            "6\n",
            "7\n",
            "8 4 10\n",
            "10 13 15\n",
            "12 2 6\n",
            "14 3 7\n",
            "i0 enable\n",
            "i1 reset\n",
            "l0 latch_Q\n",
            "o0 Q\n",
            "o1 !Q\n",
        ))
        .unwrap();

        let header = reader.header();
        assert_eq!(
            header,
            Header {
                m: 7,
                i: 2,
                l: 1,
                o: 2,
                a: 4,
                b: 0,
            }
        );

        let mut records = reader.records();
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(2)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Input(Literal(4)))));
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Latch {
                output: Literal(6),
                input: Literal(8),
                init: false
            }))
        );
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(6)))));
        assert_eq!(records.next(), Some(Ok(Aiger::Output(Literal(7)))));
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                output: Literal(8),
                inputs: [Literal(4), Literal(10)],
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                output: Literal(10),
                inputs: [Literal(13), Literal(15)],
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                output: Literal(12),
                inputs: [Literal(2), Literal(6)],
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::AndGate {
                output: Literal(14),
                inputs: [Literal(3), Literal(7)],
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 0,
                type_spec: Symbol::Input,
                symbol: "enable".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 1,
                type_spec: Symbol::Input,
                symbol: "reset".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 0,
                type_spec: Symbol::Latch,
                symbol: "latch_Q".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 0,
                type_spec: Symbol::Output,
                symbol: "Q".to_string(),
            }))
        );
        assert_eq!(
            records.next(),
            Some(Ok(Aiger::Symbol {
                position: 1,
                type_spec: Symbol::Output,
                symbol: "!Q".to_string(),
            }))
        );
        assert_eq!(records.next(), None);
    }
}
