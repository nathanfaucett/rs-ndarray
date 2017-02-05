use array::Array;
use one::One;
use zero::Zero;

use std::fmt::Debug;
use std::ops::Add;

use ndarray::NDArray;


pub struct NDArrayBuilder<T> {
    width: usize,
    height: usize,
    values: Option<Array<T>>,
}

impl<T> NDArrayBuilder<T> {

    pub fn new() -> Self {
        NDArrayBuilder {
            width: 1usize,
            height: 1usize,
            values: None,
        }

    }

    pub fn len(&self) -> usize {
        self.width * self.height
    }

    pub fn size(&mut self, width: usize, height: usize) -> &mut Self {
        self.width = width;
        self.height = height;
        self
    }
}

macro_rules! build_array {
    ($width: expr, $height: expr, $values: ident) => ({
        let mut array = Array::new($height);

        for x in 0..$height {
            array[x] = Array::new($width);
        }

        let mut x = 0;
        let mut y = 0;
        for value in $values.iter() {
            let subarray = unsafe { array.get_unchecked_mut(x) };
            subarray[y] = *value;

            if y >= $width - 1 {
                x += 1;
                y = 0;
            } else {
                y += 1;
            }
        }

        ($width, $height, array)
    })
}

impl<T> NDArrayBuilder<T>
    where T: Debug + Copy + One + Zero + Add<T, Output = T>,
{
    pub fn build_raw(&self) -> (usize, usize, Array<Array<T>>) {
        let values = match self.values {
            Some(ref values) => values.clone(),
            None => {
                let mut values = Array::new(self.len());
                let mut value = T::zero();
                for i in 0..self.len() {
                    values[i] = value;
                    value = value + T::one();
                }
                values
            },
        };

        build_array!(self.width, self.height, values)
    }

    pub fn build(&self) -> NDArray<T> {
        NDArray::new(self)
    }
}
