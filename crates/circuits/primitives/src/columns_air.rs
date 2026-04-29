use openvm_stark_backend::p3_air::BaseAir;

/// If available, returns the names of columns used in this AIR.
pub trait ColumnsAir<F>: BaseAir<F> {
    /// If the result is `Some(names)`, `names.len() == air.width()` should always
    /// be true.
    fn columns(&self) -> Option<Vec<String>> {
        None
    }
}

/// Implements [`ColumnsAir<F>`] by deferring to a [`StructReflection`](crate::StructReflection)
/// derive on the columns struct. Two forms:
///
/// ```ignore
/// // No extra generics on the AIR:
/// impl_columns_air!(MyAir, MyCols<F>);
///
/// // With const generics on the AIR (Cols generics may differ); brackets avoid
/// // ambiguity with Rust 2024 const-trait-impl syntax:
/// impl_columns_air!([const N: usize, const M: usize] MyAir<N, M>, MyCols<F, N>);
/// ```
#[macro_export]
macro_rules! impl_columns_air {
    // The bracketed-generics arm must come first; otherwise the simpler arm's `$air:ty` would
    // try to parse `[const M: usize]` as an array type.
    ([$($gp:tt)*] $air:ty, $cols:ty) => {
        impl<F: openvm_stark_backend::p3_field::Field, $($gp)*> $crate::ColumnsAir<F> for $air {
            fn columns(&self) -> Option<Vec<String>> {
                <$cols as $crate::StructReflectionHelper>::struct_reflection()
            }
        }
    };
    ($air:ty, $cols:ty) => {
        impl<F: openvm_stark_backend::p3_field::Field> $crate::ColumnsAir<F> for $air {
            fn columns(&self) -> Option<Vec<String>> {
                <$cols as $crate::StructReflectionHelper>::struct_reflection()
            }
        }
    };
}
