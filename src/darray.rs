use std::ops::*;

use array::Array;
use min::Min;


#[derive(Clone)]
pub struct DArray<T> {
    array: Array<T>,
}

impl<T> DArray<T> {
    pub fn new(size: usize) -> Self {
        DArray {
            array: Array::new(size),
        }
    }
    pub fn len(&self) -> usize {
        self.array.len()
    }
}

impl<T> Deref for DArray<T> {
    type Target = Array<T>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}
impl<T> DerefMut for DArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}


macro_rules! create_binary {
    ($Trait: ident, $func: ident, $op: tt) => (
        impl<T: Clone + $Trait<T, Output = T>> $Trait<Self> for DArray<T> {
            type Output = Self;

            fn $func(self, other: Self) -> Self {
                let len = self.len().min(other.len());
                let mut out = DArray::new(len);

                for i in 0..len {
                    out[i] = self[i].clone() $op other[i].clone();
                }
                out
            }
        }
    );
}

macro_rules! create_unary {
    ($Trait: ident, $func: ident, $op: tt) => (
        impl<T: Clone + $Trait<Output = T>> $Trait for DArray<T> {
            type Output = Self;

            fn $func(self) -> Self {
                let len = self.len();
                let mut out = DArray::new(len);

                for i in 0..len {
                    out[i] = $op(self[i].clone());
                }
                out
            }
        }
    );
}

macro_rules! create_binary_assign {
    ($Trait: ident, $func: ident, $op: tt) => (
        impl<T: Clone + $Trait<T>> $Trait<Self> for DArray<T> {
            fn $func(&mut self, other: Self) {
                let len = self.len().min(other.len());

                for i in 0..len {
                    self[i] $op other[i].clone();
                }
            }
        }
    );
}

create_binary!(Add, add, +);
create_binary!(BitAnd, bitand, &);
create_binary!(BitOr, bitor, |);
create_binary!(BitXor, bitxor, ^);
create_binary!(Div, div, /);
create_binary!(Mul, mul, *);
create_binary!(Rem, rem, %);
create_binary!(Shl, shl, <<);
create_binary!(Shr, shr, >>);
create_binary!(Sub, sub, -);

create_unary!(Neg, neg, -);
create_unary!(Not, not, !);

create_binary_assign!(AddAssign, add_assign, +=);
create_binary_assign!(BitAndAssign, bitand_assign, &=);
create_binary_assign!(BitOrAssign, bitor_assign, |=);
create_binary_assign!(BitXorAssign, bitxor_assign, ^=);
create_binary_assign!(DivAssign, div_assign, /=);
create_binary_assign!(MulAssign, mul_assign, *=);
create_binary_assign!(RemAssign, rem_assign, %=);
create_binary_assign!(ShlAssign, shl_assign, <<=);
create_binary_assign!(ShrAssign, shr_assign, >>=);
create_binary_assign!(SubAssign, sub_assign, -=);


#[cfg(test)]
mod test {
    use super::*;

    fn vec3_identity() -> DArray<isize> {
        let mut vec3 = DArray::new(3);

        vec3[0] = 1;
        vec3[1] = 1;
        vec3[2] = 1;

        vec3
    }
    fn mat3_identity() -> DArray<DArray<isize>> {
        let mut mat3 = DArray::new(3);
        mat3[0] = DArray::new(3);
        mat3[1] = DArray::new(3);
        mat3[2] = DArray::new(3);

        mat3[0][0] = 1;
        mat3[0][1] = 0;
        mat3[0][2] = 0;

        mat3[1][0] = 0;
        mat3[1][1] = 1;
        mat3[1][2] = 0;

        mat3[2][0] = 0;
        mat3[2][1] = 0;
        mat3[2][2] = 1;

        mat3
    }


    #[test]
    fn test_add_1dim() {
        let a = vec3_identity();
        let b = vec3_identity();
        let c = a + b;

        assert_eq!(c[0], 2);
        assert_eq!(c[1], 2);
        assert_eq!(c[2], 2);
    }
    #[test]
    fn test_add_2dim() {
        let a = mat3_identity();
        let b = mat3_identity();
        let c = a + b;

        assert_eq!(c[0][0], 2);
        assert_eq!(c[0][1], 0);
        assert_eq!(c[0][2], 0);

        assert_eq!(c[1][0], 0);
        assert_eq!(c[1][1], 2);
        assert_eq!(c[1][2], 0);

        assert_eq!(c[2][0], 0);
        assert_eq!(c[2][1], 0);
        assert_eq!(c[2][2], 2);
    }
}
