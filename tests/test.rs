extern crate ndarray;


use ndarray::NDArray;


macro_rules! test_trait {
    (
        $test: ident, $op: tt,
        $test_assign: ident, $assign_op: tt
    ) => (
        #[test]
        fn $test() {
            let mut a = NDArray::<usize>::new_with_shape(&[2, 2]);
            let mut b = NDArray::<usize>::new_with_shape(&[2, 2]);
            a.one();
            b.one();
            a $op b;
        }
    );
}

test_trait!(test_add, +, test_add_assign, +=);
test_trait!(test_sub, -, test_sub_assign, -=);
test_trait!(test_mul, *, test_mul_assign, *=);
test_trait!(test_div, /, test_div_assign, /=);

#[test]
fn test_ndarray() {
    let mut ndarray = NDArray::<usize>::new();

    ndarray
        .set_shape(&[3, 4])
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
        .set_shape(&[3, 3])
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
        .set_shape(&[2, 3, 4, 5, 6, 7, 8, 9, 10])
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
