use buffer::Buffer;
use one::One;
use zero::Zero;

use collections::string::String;

use core::ops::{Deref, DerefMut, Add};


pub struct NDArray<T> {
    shape: Buffer<usize>,
    multipliers: Buffer<usize>,
    data: Buffer<T>,
}

impl<T> NDArray<T> {
    pub fn new() -> Self {
        let mut shape = Buffer::new(1);
        let mut multipliers = Buffer::new(1);

        shape[0] = 1;
        multipliers[0] = 1;

        NDArray {
            shape: shape,
            multipliers: multipliers,
            data: Buffer::new(1),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.rank()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn dim(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    pub fn arange(&mut self, size: usize) -> &mut Self {
        if size > 0 && self.data.len() != size {
            self.data.resize(size);
        }
        self
    }
    pub fn reshape(&mut self, dims: &[usize]) -> &mut Self {
        if self.shape.len() != dims.len() {
            self.shape.resize(dims.len());
        }

        let mut size = 1;
        let mut i = 0;

        for value in dims.iter() {
            self.shape[i] = *value;
            size *= *value;
            i += 1;
        }

        self.recalculate_multipliers();
        self.arange(size)
    }

    fn recalculate_multipliers(&mut self) {
        let rank = self.rank();

        if self.multipliers.len() != rank {
            self.multipliers.resize(rank);
        }

        self.multipliers[rank - 1] = 1;

        for i in (0..(rank - 1)).rev() {
            self.multipliers[i] = self.multipliers[i + 1] * self.shape[i + 1];
        }
    }

    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;

        for i in 0..self.rank() {
            index += indices[i] * self.multipliers[i];
        }

        index
    }
    pub fn unravel_index(&self, index: usize) -> Buffer<usize> {
        let rank = self.rank();
        let mut indices = Buffer::new(rank);

        for i in 0..rank {
            indices[i] = (index / self.multipliers[i]) % self.shape[i];
        }

        indices
    }

    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        let index = self.ravel_index(indices);
        &self.data[index]
    }
    pub unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        let index = self.ravel_index(indices);
        &mut self.data[index]
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let index = self.ravel_index(indices);

        if index < self.len() {
            Some(&self.data[index])
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let index = self.ravel_index(indices);

        if index < self.len() {
            Some(&mut self.data[index])
        } else {
            None
        }
    }
}

impl<T> Deref for NDArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &*self.data
    }
}
impl<T> DerefMut for NDArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.data
    }
}

impl<T: Copy + One + Zero + Add<T, Output = T>> NDArray<T> {
    pub fn count(&mut self) -> &mut Self {
        let mut value = T::zero();

        for data in self.data.iter_mut() {
            *data = value;
            value = value + T::one();
        }

        self
    }
}
impl<T: Copy + Zero> NDArray<T> {
    pub fn zero(&mut self) -> &mut Self {
        for data in self.data.iter_mut() {
            *data = T::zero();
        }
        self
    }
}
impl<T: Copy + One> NDArray<T> {
    pub fn one(&mut self) -> &mut Self {
        for data in self.data.iter_mut() {
            *data = T::one();
        }
        self
    }
}
impl NDArray<String> {
    pub fn string(&mut self) -> &mut Self {
        for data in self.data.iter_mut() {
            *data = String::new();
        }
        self
    }
}


#[cfg(test)]
mod test {
    use super::*;


    #[test]
    fn test_ndarray() {
        let mut ndarray = NDArray::<usize>::new();

        ndarray
            .arange(12)
            .reshape(&[3, 4])
            .count();

        assert_eq!(ndarray.dim(0), 3);
        assert_eq!(ndarray.dim(1), 4);
        assert_eq!(ndarray.len(), 12);
        assert_eq!(ndarray.rank(), 2);

        unsafe {
            assert_eq!(ndarray.get_unchecked(&[0, 3]), &3);
            assert_eq!(ndarray.get_unchecked(&[1, 3]), &7);
            assert_eq!(ndarray.get_unchecked(&[2, 3]), &11);
        }

        assert_eq!(&*ndarray.unravel_index(3), &[0, 3]);
        assert_eq!(&*ndarray.unravel_index(7), &[1, 3]);
        assert_eq!(&*ndarray.unravel_index(11), &[2, 3]);
    }
    #[test]
    fn test_ndarray_string() {
        let mut ndarray = NDArray::<String>::new();

        ndarray
            .reshape(&[3, 3])
            .string();

        unsafe {
            assert_eq!(ndarray.get_unchecked(&[1, 1]), &"");
        }

        assert_eq!(ndarray.dim(0), 3);
        assert_eq!(ndarray.dim(1), 3);
    }

    #[test]
    fn test_deep_ndarray() {
        let mut ndarray = NDArray::<usize>::new();

        ndarray
            .reshape(&[2, 3, 4, 5, 6, 7, 8, 9, 10])
            .count();

        assert_eq!(ndarray.len(), 3628800);
        assert_eq!(ndarray.rank(), 9);

        unsafe {
            assert_eq!(ndarray.get_unchecked(&[0, 0, 0, 0, 0, 0, 0, 0, 0]), &0);
            assert_eq!(ndarray.get_unchecked(&[0, 0, 0, 2, 4, 5, 2, 5, 3]), &84473);
            assert_eq!(ndarray.get_unchecked(&[1, 0, 0, 0, 0, 0, 0, 0, 0]), &1814400);
            assert_eq!(ndarray.get_unchecked(&[1, 2, 3, 4, 5, 6, 7, 8, 9]), &3628799);
        }

        assert_eq!(&*ndarray.unravel_index(0), &[0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&*ndarray.unravel_index(84473), &[0, 0, 0, 2, 4, 5, 2, 5, 3]);
        assert_eq!(&*ndarray.unravel_index(1814400), &[1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&*ndarray.unravel_index(3628799), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
