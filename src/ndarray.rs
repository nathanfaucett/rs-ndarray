use array::Array;
use one::One;
use zero::Zero;

use collections::string::String;

use core::ops::{
    Deref, DerefMut,
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign
};


pub struct NDArray<T> {
    shape: Array<usize>,
    multipliers: Array<usize>,
    data: Array<T>,
}

impl<T> NDArray<T> {

    pub fn new() -> Self {
        Self::new_with_shape(&[1])
    }

    pub fn new_with_shape(dims: &[usize]) -> Self {
        let rank = dims.len();
        let mut shape = Array::new(rank);
        let mut multipliers = Array::new(rank);
        let size = Self::size_from_dims(dims, &mut shape);

        Self::calculate_multipliers(rank, &mut multipliers, &shape);

        NDArray {
            shape: shape,
            multipliers: multipliers,
            data: Array::new(size),
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
    pub fn shape(&self) -> &[usize] {
        &*self.shape
    }

    pub fn set_size(&mut self, size: usize) -> &mut Self {
        if self.data.len() != size {
            self.data.resize(size);
        }
        self
    }

    pub fn set_shape(&mut self, dims: &[usize]) -> &mut Self {
        if self.shape.len() != dims.len() {
            self.shape.resize(dims.len());
        }
        let size = Self::size_from_dims(dims, &mut self.shape);
        Self::calculate_multipliers(self.rank(), &mut self.multipliers, &self.shape);
        self.set_size(size)
    }

    pub fn reshape(&mut self, dims: &[usize]) -> &mut Self {
        if self.shape.len() != dims.len() {
            self.shape.resize(dims.len());
        }
        let size = Self::size_from_dims(dims, &mut self.shape);

        if size != self.data.len() {
            panic!("Invalid size from shape");
        }

        Self::calculate_multipliers(self.rank(), &mut self.multipliers, &self.shape);
        self
    }

    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;

        for i in 0..self.rank() {
            index += indices[i] * self.multipliers[i];
        }

        index
    }
    pub fn unravel_index(&self, index: usize) -> Array<usize> {
        let rank = self.rank();
        let mut indices = Array::new(rank);

        for i in 0..rank {
            indices[i] = (index / self.multipliers[i]) % self.shape[i];
        }

        indices
    }

    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        let index = self.ravel_index(indices);
        println!("{:?} {:?}", index, self.data.len());
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

    fn size_from_dims(dims: &[usize], shape: &mut Array<usize>) -> usize {
        let mut size = 1;
        let mut i = 0;

        for value in dims.iter() {
            shape[i] = *value;
            size *= *value;
            i += 1;
        }
        size
    }
    fn calculate_multipliers(
        rank: usize,
        multipliers: &mut Array<usize>,
        shape: &Array<usize>
    ) {
        if multipliers.len() != rank {
            multipliers.resize(rank);
        }

        multipliers[rank - 1] = 1;

        for i in (0..(rank - 1)).rev() {
            multipliers[i] = multipliers[i + 1] * shape[i + 1];
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


macro_rules! impl_bin_ops {
    (
        $BinTrait: ident, $bin_fn: ident, $bin_op: tt,
        $AssignTrait: ident, $assign_fn: ident, $assign_op: tt
    ) => (
        impl<'a , T: Clone + $BinTrait<T, Output = T>> $BinTrait<&'a NDArray<T>> for &'a NDArray<T> {
            type Output = NDArray<T>;

            fn $bin_fn(self, other: &'a NDArray<T>) -> Self::Output {
                let len = self.len();
                let mut out = NDArray::new_with_shape(self.shape());

                for i in 0..len {
                    out[i] = self[i].clone() $bin_op other[i].clone();
                }
                out
            }
        }
        impl<T: Clone + $BinTrait<T, Output = T>> $BinTrait<NDArray<T>> for NDArray<T> {
            type Output = NDArray<T>;

            fn $bin_fn(self, other: NDArray<T>) -> Self::Output {
                let len = self.len();
                let mut out = NDArray::new_with_shape(self.shape());

                for i in 0..len {
                    out[i] = self[i].clone() $bin_op other[i].clone();
                }
                out
            }
        }

        impl<'a , T: Clone + $AssignTrait<T>> $AssignTrait<&'a NDArray<T>> for NDArray<T> {
            fn $assign_fn(&mut self, other: &'a NDArray<T>) {
                let len = self.len();

                for i in 0..len {
                    self[i] $assign_op other[i].clone();
                }
            }
        }
        impl<'a , T: Clone + $AssignTrait<T>> $AssignTrait<NDArray<T>> for NDArray<T> {
            fn $assign_fn(&mut self, other: NDArray<T>) {
                let len = self.len();

                for i in 0..len {
                    self[i] $assign_op other[i].clone();
                }
            }
        }
    )
}


impl_bin_ops!(Add, add, +, AddAssign, add_assign, +=);
impl_bin_ops!(Sub, sub, -, SubAssign, sub_assign, -=);
impl_bin_ops!(Mul, mul, *, MulAssign, mul_assign, *=);
impl_bin_ops!(Div, div, /, DivAssign, div_assign, /=);
