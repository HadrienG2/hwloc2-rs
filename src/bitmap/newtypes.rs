//! Implementation of bitmap newtypes ([`CpuSet`] and [`NodeSet`])

#[cfg(doc)]
use super::BitmapRef;
use super::{hwloc_bitmap_s, Bitmap};
use crate::Sealed;
#[cfg(doc)]
use crate::{cpu::cpuset::CpuSet, memory::nodeset::NodeSet};
use std::{
    borrow::{Borrow, BorrowMut},
    fmt::{Debug, Display, Pointer},
    ops::Deref,
    ptr::NonNull,
};

/// A [`Bitmap`] or a specialized form thereof ([`CpuSet`], [`NodeSet`]...)
///
/// This type cannot be implemented outside of this crate as it relies on
/// hwlocality implementation details. It is only meant to be used in the
/// signature of generic methods that accept the aforementioned set of types
/// and then go on to process them homogeneously.
//
// --- Implementation details ---
//
// # Safety
//
// Implementations of this type must be a `repr(transparent)` wrapper of
// `NonNull<hwloc_bitmap_s>`, possibly with some ZSTs added.
#[allow(clippy::missing_safety_doc)]
pub unsafe trait OwnedBitmap:
    Borrow<Bitmap>
    + BorrowMut<Bitmap>
    + Clone
    + Debug
    + Display
    + From<Bitmap>
    + Into<Bitmap>
    + PartialEq
    + Pointer
    + Sealed
    + 'static
{
    /// Access the inner `NonNull<hwloc_bitmap_s>`
    ///
    /// # Safety
    ///
    /// The client must not use this pointer to mutate the target object
    #[doc(hidden)]
    unsafe fn inner(&self) -> NonNull<hwloc_bitmap_s>;
}
//
impl Sealed for Bitmap {}
//
// SAFETY: Bitmap is a repr(transparent) newtype of NonNull<RawBitmap>
unsafe impl OwnedBitmap for Bitmap {
    unsafe fn inner(&self) -> NonNull<hwloc_bitmap_s> {
        self.0
    }
}

/// An owned specialized bitmap ([`CpuSet`] or [`NodeSet`])
///
/// hwlocality avoids the need for error-prone hwloc-style `BYNODESET` flags by
/// making [`CpuSet`] and [`NodeSet`] full-blown types, not typedefs. Functions
/// which want to accept either of these specialized bitmap types can be made
/// generic over this [`SpecializedBitmap`] trait, which can be used to query
/// which specialized bitmap type was passed in and then switch to generic
/// [`Bitmap`] processing.
///
/// See also [`SpecializedBitmapRef`].
#[doc(alias = "HWLOC_MEMBIND_BYNODESET")]
#[doc(alias = "HWLOC_RESTRICT_FLAG_BYNODESET")]
pub trait SpecializedBitmap: OwnedBitmap {
    /// Tag used to discriminate between specialized bitmaps in code
    const BITMAP_KIND: BitmapKind;
}

/// A reference (typically `&T` or [`BitmapRef`]) to a [`SpecializedBitmap`]
pub trait SpecializedBitmapRef: Deref<Target = Self::Owned> {
    /// Owned form of this specialized bitmap
    type Owned: SpecializedBitmap;

    /// Construct an owned specialized bitmap from a borrowed one
    ///
    /// This is a generalization of [`ToOwned`] that supports [`BitmapRef`].
    fn to_owned(&self) -> Self::Owned {
        (**self).clone()
    }

    /// Reinterpret as a reference to a non-specialized bitmap
    fn as_bitmap_ref(&self) -> &Bitmap {
        (**self).borrow()
    }
}
//
impl<Owned: SpecializedBitmap, Ref: Deref<Target = Owned>> SpecializedBitmapRef for Ref {
    type Owned = Owned;
}

/// Kind of specialized bitmap
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum BitmapKind {
    /// This bitmap is a [`CpuSet`]
    CpuSet,

    /// This bitmap is a [`NodeSet`]
    NodeSet,
}

/// Implement a specialized bitmap
#[macro_export]
#[doc(hidden)]
macro_rules! impl_bitmap_newtype {
    (
        $(#[$attr:meta])*
        $newtype:ident
    ) => {
        impl_bitmap_newtype!(
            $(#[$attr])*
            { $newtype => bitmap_newtype }
        );
    };
    (
        $(#[$attr:meta])*
        { $newtype:ident => $mod_name:ident }
    ) => {
        #[allow(unused_imports)]
        mod $mod_name {
            use super::*;
            use $crate::{
                bitmap::{
                    Bitmap, BitmapIndex, BitmapKind, BitmapRef, OwnedBitmap,
                    Iter, SpecializedBitmap,
                },
            };
            use derive_more::{AsMut, AsRef, From, Into, IntoIterator, Not};
            use hwlocality_sys::hwloc_bitmap_s;
            #[allow(unused)]
            #[cfg(any(test, feature = "proptest"))]
            use proptest::prelude::*;
            #[cfg(test)]
            use similar_asserts::assert_eq;
            use std::{
                borrow::{Borrow, BorrowMut},
                cmp::Ordering,
                fmt::{self, Debug, Display, Formatter, Pointer},
                hash::{Hash, Hasher},
                ops::{
                    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Deref,
                    BitXorAssign, RangeBounds, Sub, SubAssign
                },
                ptr::NonNull
            };

            $(#[$attr])*
            #[derive(
                AsMut,
                AsRef,
                Clone,
                Default,
                Eq,
                From,
                Into,
                IntoIterator,
                Not,
                Ord,
            )]
            #[repr(transparent)]
            pub struct $newtype(Bitmap);

            impl SpecializedBitmap for $newtype {
                const BITMAP_KIND: BitmapKind = BitmapKind::$newtype;
            }

            /// # Re-export of the Bitmap API
            ///
            /// Only documentation headers are repeated here, you will find most of
            /// the documentation attached to identically named `Bitmap` methods.
            impl $newtype {
                /// Wraps an owned nullable hwloc bitmap
                ///
                /// See [`Bitmap::from_owned_raw_mut`](crate::bitmap::Bitmap::from_owned_raw_mut).
                #[allow(unused)]
                pub(crate) unsafe fn from_owned_raw_mut(bitmap: *mut hwloc_bitmap_s) -> Option<Self> {
                    // SAFETY: Safety contract inherited from identical Bitmap method
                    unsafe {
                        Bitmap::from_owned_raw_mut(bitmap).map(Self::from)
                    }
                }

                /// Wraps an owned hwloc bitmap
                ///
                /// See [`Bitmap::from_owned_nonnull`](crate::bitmap::Bitmap::from_owned_nonnull).
                #[allow(unused)]
                pub(crate) unsafe fn from_owned_nonnull(bitmap: NonNull<hwloc_bitmap_s>) -> Self {
                    // SAFETY: Safety contract inherited from identical Bitmap method
                    unsafe {
                        Self::from(Bitmap::from_owned_nonnull(bitmap))
                    }
                }

                /// Wraps a borrowed nullable hwloc bitmap
                ///
                /// See [`Bitmap::borrow_from_raw`](crate::bitmap::Bitmap::borrow_from_raw).
                #[allow(unused)]
                pub(crate) unsafe fn borrow_from_raw<'target>(
                    bitmap: *const hwloc_bitmap_s
                ) -> Option<BitmapRef<'target, Self>> {
                    // SAFETY: Safety contract inherited from identical Bitmap method
                    unsafe {
                        Bitmap::borrow_from_raw(bitmap).map(BitmapRef::cast)
                    }
                }

                /// Wraps a borrowed nullable hwloc bitmap
                ///
                /// See [`Bitmap::borrow_from_raw_mut`](crate::bitmap::Bitmap::borrow_from_raw_mut).
                #[allow(unused)]
                pub(crate) unsafe fn borrow_from_raw_mut<'target>(
                    bitmap: *mut hwloc_bitmap_s
                ) -> Option<BitmapRef<'target, Self>> {
                    // SAFETY: Safety contract inherited from identical Bitmap method
                    unsafe {
                        Bitmap::borrow_from_raw_mut(bitmap).map(BitmapRef::cast)
                    }
                }

                /// Wraps a borrowed hwloc bitmap
                ///
                /// See [`Bitmap::borrow_from_nonnull`](crate::bitmap::Bitmap::borrow_from_nonnull).
                #[allow(unused)]
                pub(crate) unsafe fn borrow_from_nonnull<'target>(
                    bitmap: NonNull<hwloc_bitmap_s>
                ) -> BitmapRef<'target, Self> {
                    // SAFETY: Safety contract inherited from identical Bitmap method
                    unsafe {
                        Bitmap::borrow_from_nonnull(bitmap).cast()
                    }
                }

                /// Contained bitmap pointer (for interaction with hwloc)
                ///
                /// See [`Bitmap::as_ptr`](crate::bitmap::Bitmap::as_ptr).
                #[allow(unused)]
                pub(crate) fn as_ptr(&self) -> *const hwloc_bitmap_s {
                    self.0.as_ptr()
                }

                /// Contained mutable bitmap pointer (for interaction with hwloc)
                ///
                /// See [`Bitmap::as_mut_ptr`](crate::bitmap::Bitmap::as_mut_ptr).
                #[allow(unused)]
                pub(crate) fn as_mut_ptr(&mut self) -> *mut hwloc_bitmap_s {
                    self.0.as_mut_ptr()
                }

                /// Create an empty bitmap
                ///
                /// See [`Bitmap::new`](crate::bitmap::Bitmap::new).
                pub fn new() -> Self {
                    Self::from(Bitmap::new())
                }

                /// Create a full bitmap
                ///
                /// See [`Bitmap::full`](crate::bitmap::Bitmap::full).
                pub fn full() -> Self {
                    Self::from(Bitmap::full())
                }

                /// Creates a new bitmap with the given range of indices set
                ///
                /// See [`Bitmap::from_range`](crate::bitmap::Bitmap::from_range).
                pub fn from_range<Idx>(range: impl RangeBounds<Idx>) -> Self
                where
                    Idx: Copy + TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    Self::from(Bitmap::from_range(range))
                }

                /// Turn this bitmap into a copy of another bitmap
                ///
                /// See [`Bitmap::copy_from`](crate::bitmap::Bitmap::copy_from).
                pub fn copy_from(&mut self, other: impl Deref<Target = Self>) {
                    self.0.copy_from(&other.0)
                }

                /// Clear all indices
                ///
                /// See [`Bitmap::clear`](crate::bitmap::Bitmap::clear).
                pub fn clear(&mut self) {
                    self.0.clear()
                }

                /// Set all indices
                ///
                /// See [`Bitmap::fill`](crate::bitmap::Bitmap::fill).
                pub fn fill(&mut self) {
                    self.0.fill()
                }

                /// Clear all indices except for `idx`, which is set
                ///
                /// See [`Bitmap::set_only`](crate::bitmap::Bitmap::set_only).
                pub fn set_only<Idx>(&mut self, idx: Idx)
                where
                    Idx: TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.set_only(idx)
                }

                /// Set all indices except for `idx`, which is cleared
                ///
                /// See [`Bitmap::set_all_but`](crate::bitmap::Bitmap::set_all_but).
                pub fn set_all_but<Idx>(&mut self, idx: Idx)
                where
                    Idx: TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.set_all_but(idx)
                }

                /// Set index `idx`
                ///
                /// See [`Bitmap::set`](crate::bitmap::Bitmap::set).
                pub fn set<Idx>(&mut self, idx: Idx)
                where
                    Idx: TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.set(idx)
                }

                /// Set indices covered by `range`
                ///
                /// See [`Bitmap::set_range`](crate::bitmap::Bitmap::set_range).
                pub fn set_range<Idx>(&mut self, range: impl RangeBounds<Idx>)
                where
                    Idx: Copy + TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.set_range(range)
                }

                /// Clear index `idx`
                ///
                /// See [`Bitmap::unset`](crate::bitmap::Bitmap::unset).
                pub fn unset<Idx>(&mut self, idx: Idx)
                where
                    Idx: TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.unset(idx)
                }

                /// Clear indices covered by `range`
                ///
                /// See [`Bitmap::unset_range`](crate::bitmap::Bitmap::unset_range).
                pub fn unset_range<Idx>(&mut self, range: impl RangeBounds<Idx>)
                where
                    Idx: Copy + TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.unset_range(range)
                }

                /// Keep a single index among those set in the bitmap
                ///
                /// See [`Bitmap::singlify`](crate::bitmap::Bitmap::singlify).
                pub fn singlify(&mut self) {
                    self.0.singlify()
                }

                /// Check if index `idx` is set
                ///
                /// See [`Bitmap::is_set`](crate::bitmap::Bitmap::is_set).
                pub fn is_set<Idx>(&self, idx: Idx) -> bool
                where
                    Idx: TryInto<BitmapIndex>,
                    <Idx as TryInto<BitmapIndex>>::Error: Debug,
                {
                    self.0.is_set(idx)
                }

                /// Check if all indices are unset
                ///
                /// See [`Bitmap::is_empty`](crate::bitmap::Bitmap::is_empty).
                pub fn is_empty(&self) -> bool {
                    self.0.is_empty()
                }

                /// Check if all indices are set
                ///
                /// See [`Bitmap::is_full`](crate::bitmap::Bitmap::is_full).
                pub fn is_full(&self) -> bool {
                    self.0.is_full()
                }

                /// Check the first set index, if any
                ///
                /// See [`Bitmap::first_set`](crate::bitmap::Bitmap::first_set).
                pub fn first_set(&self) -> Option<BitmapIndex> {
                    self.0.first_set()
                }

                /// Iterate over set indices
                ///
                /// See [`Bitmap::iter_set`](crate::bitmap::Bitmap::iter_set).
                pub fn iter_set(&self) -> Iter<&Bitmap> {
                    self.0.iter_set()
                }

                /// Check the last set index, if any
                ///
                /// See [`Bitmap::last_set`](crate::bitmap::Bitmap::last_set).
                pub fn last_set(&self) -> Option<BitmapIndex> {
                    self.0.last_set()
                }

                /// The number of indices that are set in the bitmap
                ///
                /// See [`Bitmap::weight`](crate::bitmap::Bitmap::weight).
                pub fn weight(&self) -> Option<usize> {
                    self.0.weight()
                }

                /// Check the first unset index, if any
                ///
                /// See [`Bitmap::first_unset`](crate::bitmap::Bitmap::first_unset).
                pub fn first_unset(&self) -> Option<BitmapIndex> {
                    self.0.first_unset()
                }

                /// Iterate over unset indices
                ///
                /// See [`Bitmap::iter_unset`](crate::bitmap::Bitmap::iter_unset).
                pub fn iter_unset(&self) -> Iter<&Bitmap> {
                    self.0.iter_unset()
                }

                /// Check the last unset index, if any
                ///
                /// See [`Bitmap::last_unset`](crate::bitmap::Bitmap::last_unset).
                pub fn last_unset(&self) -> Option<BitmapIndex> {
                    self.0.last_unset()
                }

                /// Optimized version of `*self = !self`
                ///
                /// See [`Bitmap::invert`](crate::bitmap::Bitmap::invert).
                pub fn invert(&mut self) {
                    self.0.invert()
                }

                /// Truth that `self` and `rhs` have some set indices in common
                ///
                /// See [`Bitmap::intersects`](crate::bitmap::Bitmap::intersects).
                pub fn intersects(&self, rhs: impl Deref<Target = Self>) -> bool {
                    self.0.intersects(&rhs.0)
                }

                /// Truth that the indices set in `inner` are a subset of those set
                /// in `self`
                ///
                /// See [`Bitmap::includes`](crate::bitmap::Bitmap::includes).
                pub fn includes(&self, inner: impl Deref<Target = Self>) -> bool {
                    self.0.includes(&inner.0)
                }
            }

            #[cfg(any(test, feature = "proptest"))]
            impl Arbitrary for $newtype {
                type Parameters = <Bitmap as Arbitrary>::Parameters;
                type Strategy = prop::strategy::Map<
                    <Bitmap as Arbitrary>::Strategy,
                    fn(Bitmap) -> Self,
                >;

                fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
                    <Bitmap as Arbitrary>::arbitrary_with(args).prop_map(Self)
                }
            }

            impl<B: Borrow<$newtype>> BitAnd<B> for &$newtype {
                type Output = $newtype;

                fn bitand(self, rhs: B) -> $newtype {
                    $newtype((&self.0) & (&rhs.borrow().0))
                }
            }

            impl<B: Borrow<Self>> BitAnd<B> for $newtype {
                type Output = Self;

                fn bitand(self, rhs: B) -> Self {
                    $newtype(self.0 & (&rhs.borrow().0))
                }
            }

            impl<B: Borrow<Self>> BitAndAssign<B> for $newtype {
                fn bitand_assign(&mut self, rhs: B) {
                    self.0 &= (&rhs.borrow().0)
                }
            }

            impl<B: Borrow<$newtype>> BitOr<B> for &$newtype {
                type Output = $newtype;

                fn bitor(self, rhs: B) -> $newtype {
                    $newtype(&self.0 | &rhs.borrow().0)
                }
            }

            impl<B: Borrow<Self>> BitOr<B> for $newtype {
                type Output = Self;

                fn bitor(self, rhs: B) -> Self {
                    $newtype(self.0 | &rhs.borrow().0)
                }
            }

            impl<B: Borrow<Self>> BitOrAssign<B> for $newtype {
                fn bitor_assign(&mut self, rhs: B) {
                    self.0 |= &rhs.borrow().0
                }
            }

            impl<B: Borrow<$newtype>> BitXor<B> for &$newtype {
                type Output = $newtype;

                fn bitxor(self, rhs: B) -> $newtype {
                    $newtype(&self.0 ^ &rhs.borrow().0)
                }
            }

            impl<B: Borrow<Self>> BitXor<B> for $newtype {
                type Output = Self;

                fn bitxor(self, rhs: B) -> Self {
                    $newtype(self.0 ^ &rhs.borrow().0)
                }
            }

            impl<B: Borrow<Self>> BitXorAssign<B> for $newtype {
                fn bitxor_assign(&mut self, rhs: B) {
                    self.0 ^= &rhs.borrow().0
                }
            }

            impl<'target> Borrow<Bitmap> for $newtype {
                fn borrow(&self) -> &Bitmap {
                    self.as_ref()
                }
            }

            impl<'target> BorrowMut<Bitmap> for $newtype {
                fn borrow_mut(&mut self) -> &mut Bitmap {
                    self.as_mut()
                }
            }

            impl Debug for $newtype {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    let text = format!("{}({:?})", stringify!($newtype), &self.0);
                    f.pad(&text)
                }
            }

            impl Display for $newtype {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    let text = format!("{}({})", stringify!($newtype), &self.0);
                    f.pad(&text)
                }
            }

            impl<BI: Borrow<BitmapIndex>> Extend<BI> for $newtype {
                fn extend<T: IntoIterator<Item = BI>>(&mut self, iter: T) {
                    self.0.extend(iter)
                }
            }

            impl<BI: Borrow<BitmapIndex>> From<BI> for $newtype {
                fn from(value: BI) -> Self {
                    Self(value.into())
                }
            }

            impl<BI: Borrow<BitmapIndex>> FromIterator<BI> for $newtype {
                fn from_iter<I: IntoIterator<Item = BI>>(iter: I) -> Self {
                    Self(Bitmap::from_iter(iter))
                }
            }

            impl Hash for $newtype {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    self.0.hash(state)
                }
            }

            impl<'newtype> IntoIterator for &'newtype $newtype {
                type Item = BitmapIndex;
                type IntoIter = Iter<&'newtype Bitmap>;

                fn into_iter(self) -> Self::IntoIter {
                    (&self.0).into_iter()
                }
            }

            impl Not for &$newtype {
                type Output = $newtype;

                fn not(self) -> $newtype {
                    $newtype(!&self.0)
                }
            }

            // SAFETY: $newtype is a repr(transparent) newtype of Bitmap, which is
            //         itself a repr(transparent) newtype of RawBitmap.
            unsafe impl OwnedBitmap for $newtype {
                unsafe fn inner(&self) -> NonNull<hwloc_bitmap_s> {
                    // SAFETY: Safety proof inherited from `Bitmap`
                    unsafe { self.0.inner() }
                }
            }

            impl<B: Borrow<Self>> PartialEq<B> for $newtype {
                fn eq(&self, other: &B) -> bool {
                    self.0 == other.borrow().0
                }
            }

            impl<B: Borrow<Self>> PartialOrd<B> for $newtype {
                fn partial_cmp(&self, other: &B) -> Option<Ordering> {
                    self.0.partial_cmp(&other.borrow().0)
                }
            }

            impl Pointer for $newtype {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    <Bitmap as Pointer>::fmt(&self.0, f)
                }
            }

            impl $crate::Sealed for $newtype {}

            impl<B: Borrow<$newtype>> Sub<B> for &$newtype {
                type Output = $newtype;

                fn sub(self, rhs: B) -> $newtype {
                    $newtype(&self.0 - &rhs.borrow().0)
                }
            }

            impl<B: Borrow<Self>> Sub<B> for $newtype {
                type Output = Self;

                fn sub(self, rhs: B) -> Self {
                    $newtype(self.0 - &rhs.borrow().0)
                }
            }

            impl<B: Borrow<Self>> SubAssign<B> for $newtype {
                fn sub_assign(&mut self, rhs: B) {
                    self.0 -= &rhs.borrow().0
                }
            }

            $crate::impl_bitmap_newtype_ref!($newtype);

            #[allow(
                clippy::cognitive_complexity,
                clippy::op_ref,
                clippy::too_many_lines,
                non_camel_case_name,
                trivial_casts
            )]
            #[cfg(test)]
            mod tests {
                use super::*;
                use $crate::{
                    bitmap::{
                        tests::{index_range, index_vec, INFINITE_EXPLORE_ITERS},
                        SpecializedBitmap, SpecializedBitmapRef,
                    },
                    strategies::bitmap_index,
                };
                #[allow(unused)]
                use similar_asserts::assert_eq;
                use std::{
                    collections::hash_map::RandomState,
                    error::Error,
                    fmt::{
                        Binary, LowerExp, LowerHex, Octal, Pointer, UpperExp, UpperHex,
                    },
                    hash::BuildHasher,
                    io::{self, Read},
                    mem::ManuallyDrop,
                    ops::Drop,
                    panic::UnwindSafe,
                    ptr,
                };
                use static_assertions::{assert_impl_all, assert_not_impl_any};

                $crate::impl_bitmap_newtype_ref_tests!($newtype);

                // Check that newtypes keep implementing all expected traits,
                // in the interest of detecting future semver-breaking changes
                assert_impl_all!($newtype:
                    AsMut<Bitmap>,
                    BitAnd<$newtype>, BitAnd<&'static $newtype>,
                    BitAndAssign<$newtype>, BitAndAssign<&'static $newtype>,
                    BitOr<$newtype>, BitOr<&'static $newtype>,
                    BitOrAssign<$newtype>, BitOrAssign<&'static $newtype>,
                    BitXor<$newtype>, BitXor<&'static $newtype>,
                    BitXorAssign<$newtype>, BitXorAssign<&'static $newtype>,
                    BorrowMut<Bitmap>, Clone, Debug, Default, Display, Eq,
                    Extend<BitmapIndex>, Extend<&'static BitmapIndex>,
                    From<Bitmap>, From<BitmapIndex>, From<&'static BitmapIndex>,
                    FromIterator<BitmapIndex>, FromIterator<&'static BitmapIndex>,
                    Hash, Into<Bitmap>, IntoIterator<Item=BitmapIndex>, Not, Ord,
                    SpecializedBitmap,
                    PartialEq<&'static $newtype>,
                    PartialOrd<&'static $newtype>,
                    Pointer, Sized,
                    Sub<$newtype>, Sub<&'static $newtype>,
                    SubAssign<$newtype>, SubAssign<&'static $newtype>,
                    Sync, Unpin, UnwindSafe
                );
                assert_not_impl_any!($newtype:
                    Binary, Copy, Deref, Error, LowerExp, LowerHex, Octal, Read,
                    UpperExp, UpperHex, fmt::Write, io::Write
                );
                assert_impl_all!(&$newtype:
                    BitAnd<$newtype>, BitAnd<&'static $newtype>,
                    BitOr<$newtype>, BitOr<&'static $newtype>,
                    BitXor<$newtype>, BitXor<&'static $newtype>,
                    IntoIterator<Item=BitmapIndex>,
                    Not<Output=$newtype>,
                    SpecializedBitmapRef,
                    Sub<$newtype>, Sub<&'static $newtype>,
                );

                #[test]
                fn static_checks() {
                    assert_eq!($newtype::BITMAP_KIND, BitmapKind::$newtype);
                }

                #[test]
                fn nullary() {
                    assert_eq!($newtype::new(), $newtype(Bitmap::new()));
                    assert_eq!($newtype::full(), $newtype(Bitmap::full()));
                    assert_eq!($newtype::default(), $newtype(Bitmap::default()));

                    // SAFETY: All of the following are safe on null input
                    unsafe {
                        assert_eq!($newtype::from_owned_raw_mut(ptr::null_mut()), None);
                        assert_eq!($newtype::borrow_from_raw(ptr::null()), None);
                        assert_eq!($newtype::borrow_from_raw_mut(ptr::null_mut()), None);
                    }
                }

                proptest! {
                    #[test]
                    fn from_idx(idx in bitmap_index()) {
                        prop_assert_eq!($newtype::from(idx), $newtype(Bitmap::from(idx)));
                    }

                    #[test]
                    fn from_range(range in index_range()) {
                        prop_assert_eq!(
                            $newtype::from_range(range.clone()),
                            $newtype(Bitmap::from_range(range))
                        );
                    }

                    #[test]
                    fn from_iterator(v in index_vec()) {
                        let new = v.iter().copied().collect::<$newtype>();
                        let inner = v.into_iter().collect::<Bitmap>();
                        prop_assert_eq!(new, $newtype(inner));
                    }

                    #[test]
                    fn to_from_bitmap(bitmap: Bitmap) {
                        let new = $newtype::from(bitmap.clone());
                        prop_assert_eq!(&new.0, &bitmap);
                        prop_assert_eq!(&Bitmap::from(new), &bitmap);
                    }
                }

                proptest! {
                    #[test]
                    fn unary(new: $newtype) {
                        // Test unary newtype operations
                        prop_assert_eq!(new.first_set(), new.0.first_set());
                        prop_assert_eq!(new.last_set(), new.0.last_set());
                        prop_assert_eq!(new.weight(), new.0.weight());
                        prop_assert_eq!(new.first_unset(), new.0.first_unset());
                        prop_assert_eq!(new.last_unset(), new.0.last_unset());
                        prop_assert_eq!(
                            format!("{new:?}"),
                            format!("{}({:?})", stringify!($newtype), new.0)
                        );
                        prop_assert_eq!(format!("{:p}", new), format!("{:p}", new.0));
                        prop_assert_eq!(
                            new.to_string(),
                            format!("{}({})", stringify!($newtype), new.0)
                        );
                        let state = RandomState::new();
                        prop_assert_eq!(state.hash_one(&new), state.hash_one(&new.0));
                        // SAFETY: No mutation going on
                        unsafe { prop_assert_eq!(new.inner(), new.0.inner()) };
                        //
                        prop_assert!(new
                            .iter_set()
                            .take(INFINITE_EXPLORE_ITERS)
                            .eq(new.0.iter_set().take(INFINITE_EXPLORE_ITERS)));
                        prop_assert!((&new)
                            .into_iter()
                            .take(INFINITE_EXPLORE_ITERS)
                            .eq(new.0.iter_set().take(INFINITE_EXPLORE_ITERS)));
                        prop_assert!(new
                            .clone()
                            .into_iter()
                            .take(INFINITE_EXPLORE_ITERS)
                            .eq(new.0.iter_set().take(INFINITE_EXPLORE_ITERS)));
                        prop_assert!(new
                            .iter_unset()
                            .take(INFINITE_EXPLORE_ITERS)
                            .eq(new.0.iter_unset().take(INFINITE_EXPLORE_ITERS)));
                        //
                        let mut buf = new.clone();
                        buf.clear();
                        prop_assert!(buf.is_empty());
                        //
                        buf.fill();
                        prop_assert!(buf.is_full());
                        //
                        if new.weight().unwrap_or(usize::MAX) >= 1 {
                            buf.copy_from(&new);
                            buf.singlify();
                            prop_assert_eq!(buf.weight(), Some(1));
                        }
                        //
                        buf.copy_from(&new);
                        buf.invert();
                        prop_assert_eq!(&buf, &$newtype(!&new.0));

                        // Test AsRef-like conversions to Bitmap and BitmapRef
                        prop_assert!(
                            ptr::eq(
                                <$newtype as AsRef<Bitmap>>::as_ref(&new),
                                &new.0
                            )
                        );
                        prop_assert!(
                            ptr::eq(
                                <$newtype as Borrow<Bitmap>>::borrow(&new),
                                &new.0
                            )
                        );
                        buf.copy_from(&new);
                        prop_assert_eq!(
                            <$newtype as AsMut<Bitmap>>::as_mut(&mut buf).as_ptr(),
                            buf.0.as_ptr()
                        );
                        prop_assert_eq!(
                            <$newtype as BorrowMut<Bitmap>>::borrow_mut(&mut buf).as_ptr(),
                            buf.0.as_ptr()
                        );

                        // Test SpecializedBitmapRef operations
                        prop_assert_eq!(&SpecializedBitmapRef::to_owned(&&new), &new);
                        prop_assert_eq!(&SpecializedBitmapRef::to_owned(&BitmapRef::from(&new)), &new);

                        // Test low-level functions and BitmapRef<$newtype>
                        test_newtype_ref_unary(&&new, BitmapRef::from(&new))?;
                        let new = ManuallyDrop::new(new);
                        let new_const = new.as_ptr();
                        let new_mut = new_const.cast_mut();
                        let new_nonnull = NonNull::new(new_mut).unwrap();
                        // SAFETY: We won't use this pointer to mutate
                        prop_assert_eq!(unsafe { new.inner() }, new_nonnull);
                        {
                            // SAFETY: If it worked for the original newtype, it works here too,
                            //         as long as we leave the original aside and refrain from
                            //         dropping the newtype on either side.
                            let new = unsafe { $newtype::from_owned_nonnull(new_nonnull) };
                            prop_assert_eq!(new.as_ptr(), new_const);
                            std::mem::forget(new);
                        }
                        {
                            // SAFETY: If it worked for the original newtype, it works here too,
                            //         as long as we leave the original aside and refrain from
                            //         dropping the newtype on either side.
                            let new = unsafe { $newtype::from_owned_raw_mut(new_nonnull.as_ptr()).unwrap() };
                            prop_assert_eq!(new.as_ptr(), new_const);
                            std::mem::forget(new);
                        }
                        let new = ManuallyDrop::into_inner(new);
                        {
                            // SAFETY: Safe as long as we don't invalidate the original
                            let borrow = unsafe { $newtype::borrow_from_nonnull(new_nonnull) };
                            prop_assert_eq!(borrow.as_ptr(), new_const);
                            test_newtype_ref_unary(&new, borrow)?;
                        }
                        {
                            // SAFETY: Safe as long as we don't invalidate the original
                            let borrow = unsafe { $newtype::borrow_from_raw(new.as_ptr()).unwrap() };
                            prop_assert_eq!(borrow.as_ptr(), new_const);
                            test_newtype_ref_unary(&new, borrow)?;
                        }
                        let mut new = new;
                        {
                            // SAFETY: Safe as long as we don't invalidate the original
                            let borrow = unsafe { $newtype::borrow_from_raw_mut(new.as_mut_ptr()).unwrap() };
                            prop_assert_eq!(borrow.as_ptr(), new_const);
                            test_newtype_ref_unary(&new, borrow)?;
                        }
                    }

                    #[test]
                    fn op_idx(
                        new: $newtype,
                        idx in bitmap_index()
                    ) {
                        let mut buf = new.clone();
                        buf.set_only(idx);
                        prop_assert_eq!(&buf, &$newtype::from(idx));
                        buf.set_all_but(idx);
                        prop_assert_eq!(&buf, &!$newtype::from(idx));

                        buf.copy_from(&new);
                        buf.set(idx);
                        prop_assert!(buf.is_set(idx));
                        let mut bitmap_buf = new.clone().0;
                        bitmap_buf.set(idx);
                        prop_assert_eq!(&buf, &$newtype::from(bitmap_buf.clone()));

                        buf.copy_from(&new);
                        buf.unset(idx);
                        prop_assert!(!buf.is_set(idx));
                        bitmap_buf.copy_from(&new.0);
                        bitmap_buf.unset(idx);
                        prop_assert_eq!(buf, $newtype::from(bitmap_buf.clone()));
                    }

                    #[test]
                    fn op_range(
                        new: $newtype,
                        range in index_range()
                    ) {
                        let mut buf = new.clone();
                        let mut bitmap_buf = new.0.clone();
                        buf.set_range(range.clone());
                        bitmap_buf.set_range(range.clone());
                        prop_assert_eq!(&buf, &$newtype(bitmap_buf.clone()));

                        buf.copy_from(&new);
                        bitmap_buf.copy_from(&new.0);
                        buf.unset_range(range.clone());
                        bitmap_buf.unset_range(range);
                        prop_assert_eq!(buf, $newtype::from(bitmap_buf));
                    }

                    #[test]
                    fn op_iterator(
                        mut new: $newtype,
                        indices in index_vec()
                    ) {
                        let mut inner = new.0.clone();
                        new.extend(indices.clone());
                        inner.extend(indices);
                        prop_assert_eq!(new, $newtype(inner));
                    }

                    #[test]
                    fn binary([new1, new2]: [$newtype; 2]) {
                        // Test binary newtype operations
                        prop_assert_eq!(new1.intersects(&new2), new1.0.intersects(&new2.0));
                        prop_assert_eq!(new1.includes(&new2), new1.0.includes(&new2.0));
                        prop_assert_eq!(&new1 & &new2, $newtype(&new1.0 & &new2.0));
                        prop_assert_eq!(&new1 | &new2, $newtype(&new1.0 | &new2.0));
                        prop_assert_eq!(&new1 ^ &new2, $newtype(&new1.0 ^ &new2.0));
                        prop_assert_eq!(&new1 - &new2, $newtype(&new1.0 - &new2.0));
                        prop_assert_eq!(new1.clone() & &new2, $newtype(&new1.0 & &new2.0));
                        prop_assert_eq!(new1.clone() | &new2, $newtype(&new1.0 | &new2.0));
                        prop_assert_eq!(new1.clone() ^ &new2, $newtype(&new1.0 ^ &new2.0));
                        prop_assert_eq!(new1.clone() - &new2, $newtype(&new1.0 - &new2.0));
                        prop_assert_eq!(&new1 == &new2, &new1 == &new2);
                        prop_assert_eq!(new1.partial_cmp(&new2), new1.0.partial_cmp(&new2.0));
                        prop_assert_eq!(new1.cmp(&new2), new1.0.cmp(&new2.0));
                        //
                        let mut buf = new1.clone();
                        buf.copy_from(&new2);
                        prop_assert_eq!(&buf, &new2);
                        //
                        buf.copy_from(&new1);
                        buf &= &new2;
                        prop_assert_eq!(&buf, &(&new1 & &new2));
                        buf.copy_from(&new1);
                        buf |= &new2;
                        prop_assert_eq!(&buf, &(&new1 | &new2));
                        buf.copy_from(&new1);
                        buf ^= &new2;
                        prop_assert_eq!(&buf, &(&new1 ^ &new2));
                        buf.copy_from(&new1);
                        buf -= &new2;
                        prop_assert_eq!(buf, &new1 - &new2);

                        // Test binary BitmapRef operations
                        test_newtype_ref_binops([new1, new2])?;
                    }
                }
            }
        }

        #[doc(inline)]
        pub use $mod_name::$newtype;
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};
    use std::{
        fmt::{self, Binary, LowerExp, LowerHex, Octal, Pointer, UpperExp, UpperHex},
        hash::Hash,
        io::{self, Read},
        panic::UnwindSafe,
    };

    // Check that public types in this module keep implementing all expected
    // traits, in the interest of detecting future semver-breaking changes
    assert_impl_all!(BitmapKind:
        Copy, Debug, Hash, Sized, Sync, Unpin, UnwindSafe
    );
    assert_not_impl_any!(BitmapKind:
        Binary, Default, Deref, Display, Drop, IntoIterator, LowerExp, LowerHex,
        Octal, PartialOrd, Pointer, Read, UpperExp, UpperHex, fmt::Write,
        io::Write
    );
}
