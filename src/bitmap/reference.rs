//! References to a [`Bitmap`]-like `Target` that is owned by hwloc

use super::{Bitmap, BitmapIndex, Iter, OwnedBitmap};
use crate::Sealed;
use hwlocality_sys::hwloc_bitmap_s;
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter, Pointer},
    hash::{self, Hash},
    marker::PhantomData,
    ops::{BitAnd, BitOr, BitXor, Deref, Not, Sub},
    ptr::NonNull,
};

/// Read-only reference to a [`Bitmap`]-like `Target` that is owned by hwloc
///
/// For most intents and purposes, you can think of this as an
/// `&'target Target` and use it as such. But it cannot literally be an
/// `&'target Target` due to annoying details of the underlying C API.
//
// --- Implementation details ---
//
// The reason why `BitmapRef` needs to be a thing is that...
//
// - In the underlying hwloc API, everything is done using `hwloc_bitmap_t`,
//   which is actually a pointer to `hwloc_bitmap_s`, that is itself an opaque
//   incomplete type that is modeled on the Rust side via `hwloc_bitmap_s`.
// - We can't directly expose `hwloc_bitmap_s` to user code because if they move it
//   around it won't do what they want (and if you have an `&mut` or an owned
//   value you can always move in Rust). Therefore, hwlocality functions that
//   return bitmaps like hwloc would return an `hwloc_bitmap_t` cannot return
//   standard Rust pointers/refs like `&hwloc_bitmap_s`, `&mut hwloc_bitmap_s`,
//   `Box<hwloc_bitmap_s>`, they must instead return some kind of
//   `NonNull<hwloc_bitmap_s>` wrapper that implements the bitmap API without
//   exposing the inner hwloc_bitmap_s.
// - We need two such wrappers because sometimes we need to return an owned
//   bitmap that must be liberated on Drop, and sometimes we need to return a
//   borrowed bitmap that must not outlive its parent.
//
// Technically, we could also have a `BitmapMut` that models `&mut hwloc_bitmap_s`,
// but so far the need for this has not materialized.
//
// # Safety
//
// - As a type invariant, the inner pointer is assumed to always point to a
//   valid, non-aliased bitmap
// - BitmapRef<Target> must never expose any operation that allows mutating the
//   target because it is constructible from &Target, if the need for mutation
//   arises we will need to add BitmapMut<Target>
#[repr(transparent)]
#[derive(Clone)]
pub struct BitmapRef<'target, Target: OwnedBitmap>(
    NonNull<hwloc_bitmap_s>,
    PhantomData<&'target Target>,
);

impl<'target, Target: OwnedBitmap> BitmapRef<'target, Target> {
    /// Wrap a borrowed hwloc bitmap
    ///
    /// # Safety
    ///
    /// The pointer must target a bitmap that is valid for `'target`. This
    /// bitmap will not be automatically freed on `Drop`.
    pub(crate) unsafe fn from_nonnull(bitmap: NonNull<hwloc_bitmap_s>) -> Self {
        // SAFETY: Per function input precondition
        Self(bitmap, PhantomData)
    }

    /// Make a copy of the target bitmap
    ///
    /// Rust references are special in that `ref.clone()` calls the [`Clone`]
    /// implementation of the pointee, and not that of the reference.
    /// [`BitmapRef`] does not enjoy this magic, and thus needs to expose target
    /// cloning via a dedicated method.
    pub fn clone_target(&self) -> Target {
        self.as_ref().clone()
    }

    /// Cast to another bitmap newtype
    pub(crate) fn cast<Other: OwnedBitmap>(self) -> BitmapRef<'target, Other> {
        BitmapRef(self.0, PhantomData)
    }
}

impl<'target, Target: OwnedBitmap> AsRef<Target> for BitmapRef<'target, Target> {
    fn as_ref(&self) -> &Target {
        // SAFETY: - Both Target and BitmapRef are effectively repr(transparent)
        //           newtypes of NonNull<hwloc_bitmap_s>, so &Target and &BitmapRef are
        //           effectively the same thing after compilation.
        //         - The borrow checker ensures that one cannot construct an
        //           &'a BitmapRef<'target> which does not verify 'target: 'a,
        //           so one cannot use this AsRef impl to build an excessively
        //           long-lived &'a Target.
        unsafe {
            let ptr: *const Self = self;
            &*ptr.cast::<Target>()
        }
    }
}

impl<Target, Rhs> BitAnd<Rhs> for &BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: BitAnd<&'b Target, Output = Target>,
{
    type Output = Target;

    fn bitand(self, rhs: Rhs) -> Target {
        self.as_ref() & rhs.borrow()
    }
}

impl<Target, Rhs> BitAnd<Rhs> for BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: BitAnd<&'b Target, Output = Target>,
{
    type Output = Target;

    fn bitand(self, rhs: Rhs) -> Target {
        self.as_ref() & rhs.borrow()
    }
}

impl<Target, Rhs> BitOr<Rhs> for &BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: BitOr<&'b Target, Output = Target>,
{
    type Output = Target;

    fn bitor(self, rhs: Rhs) -> Target {
        self.as_ref() | rhs.borrow()
    }
}

impl<Target, Rhs> BitOr<Rhs> for BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: BitOr<&'b Target, Output = Target>,
{
    type Output = Target;

    fn bitor(self, rhs: Rhs) -> Target {
        self.as_ref() | rhs.borrow()
    }
}

impl<Target, Rhs> BitXor<Rhs> for &BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: BitXor<&'b Target, Output = Target>,
{
    type Output = Target;

    fn bitxor(self, rhs: Rhs) -> Target {
        self.as_ref() ^ rhs.borrow()
    }
}

impl<Target, Rhs> BitXor<Rhs> for BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: BitXor<&'b Target, Output = Target>,
{
    type Output = Target;

    fn bitxor(self, rhs: Rhs) -> Target {
        self.as_ref() ^ rhs.borrow()
    }
}

// NOTE: This seemingly useless impl is needed in order to have impls of
//       BitXyz<&BitmapRef> for Target
#[doc(hidden)]
impl<Target: OwnedBitmap> Borrow<Target> for &BitmapRef<'_, Target> {
    fn borrow(&self) -> &Target {
        self.as_ref()
    }
}

impl<Target: OwnedBitmap> Borrow<Target> for BitmapRef<'_, Target> {
    fn borrow(&self) -> &Target {
        self.as_ref()
    }
}

// NOTE: `Target` cannot implement `Borrow<BitmapRef<'target, Target>>` because
//       it would have to do so for all `'target` which is not correct: you
//       should not be able to create, say, a `BitmapRef<'static, Bitmap>` from
//       a `&'a Bitmap` of finite lifetime, as otherwise you would be able
//       to clone this `BitmapRef`, drop the original `Bitmap`, and then you
//       would have a dangling `BitmapRef`, resulting in UB.

impl<Target: OwnedBitmap> Copy for BitmapRef<'_, Target> {}

impl<Target: OwnedBitmap + Debug> Debug for BitmapRef<'_, Target> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Target as Debug>::fmt(self.as_ref(), f)
    }
}

impl<Target: OwnedBitmap> Deref for BitmapRef<'_, Target> {
    type Target = Target;

    fn deref(&self) -> &Target {
        self.as_ref()
    }
}

impl<Target: OwnedBitmap + Display> Display for BitmapRef<'_, Target> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Target as Display>::fmt(self.as_ref(), f)
    }
}

impl<Target: OwnedBitmap + Eq + PartialEq<Self>> Eq for BitmapRef<'_, Target> {}

impl<'target, Target: OwnedBitmap> From<&'target Target> for BitmapRef<'target, Target> {
    fn from(input: &'target Target) -> Self {
        // SAFETY: A `BitmapRef` does not allow mutating the target object
        Self(unsafe { input.inner() }, PhantomData)
    }
}

impl<'target, Target: OwnedBitmap + Hash> Hash for BitmapRef<'target, Target> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl<'target, 'self_, Target> IntoIterator for &'self_ BitmapRef<'target, Target>
where
    'target: 'self_,
    Target: OwnedBitmap,
    &'self_ Target: Borrow<Bitmap>,
{
    type Item = BitmapIndex;
    type IntoIter = Iter<&'self_ Target>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.as_ref(), Bitmap::next_set)
    }
}

impl<'target, Target: OwnedBitmap> IntoIterator for BitmapRef<'target, Target> {
    type Item = BitmapIndex;
    type IntoIter = Iter<BitmapRef<'target, Bitmap>>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.cast(), Bitmap::next_set)
    }
}

impl<Target> Not for &BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    for<'target> &'target Target: Not<Output = Target>,
{
    type Output = Target;

    fn not(self) -> Target {
        !(self.as_ref())
    }
}

impl<Target> Not for BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    for<'target> &'target Target: Not<Output = Target>,
{
    type Output = Target;

    fn not(self) -> Target {
        !(self.as_ref())
    }
}

impl<Target: OwnedBitmap + Ord + PartialOrd<Self>> Ord for BitmapRef<'_, Target> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<Target, Rhs> PartialEq<Rhs> for BitmapRef<'_, Target>
where
    Target: OwnedBitmap + PartialEq<Rhs>,
{
    fn eq(&self, other: &Rhs) -> bool {
        self.as_ref() == other
    }
}

impl<Target, Rhs> PartialOrd<Rhs> for BitmapRef<'_, Target>
where
    Target: OwnedBitmap + PartialOrd<Rhs>,
{
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl<Target: OwnedBitmap> Pointer for BitmapRef<'_, Target> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <NonNull<hwloc_bitmap_s> as fmt::Pointer>::fmt(&self.0, f)
    }
}

impl<Target: OwnedBitmap> Sealed for BitmapRef<'_, Target> {}

// SAFETY: BitmapRef exposes no internal mutability
unsafe impl<Target: OwnedBitmap + Sync> Send for BitmapRef<'_, Target> {}

impl<Target, Rhs> Sub<Rhs> for &BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: Sub<&'b Target, Output = Target>,
{
    type Output = Target;

    fn sub(self, rhs: Rhs) -> Target {
        self.as_ref() - rhs.borrow()
    }
}

impl<Target, Rhs> Sub<Rhs> for BitmapRef<'_, Target>
where
    Target: OwnedBitmap,
    Rhs: Borrow<Target>,
    for<'a, 'b> &'a Target: Sub<&'b Target, Output = Target>,
{
    type Output = Target;

    fn sub(self, rhs: Rhs) -> Target {
        self.as_ref() - rhs.borrow()
    }
}

// SAFETY: BitmapRef exposes no internal mutability
unsafe impl<Target: OwnedBitmap + Sync> Sync for BitmapRef<'_, Target> {}

// NOTE: `BitmapRef` cannot implement `ToOwned` because it would require
//       `Target` to implement `Borrow<BitmapRef<'target, Target>>`, which is
//       wrong as outlined above.

/// Implement BitmapRef for a specialized bitmap
#[macro_export]
#[doc(hidden)]
macro_rules! impl_bitmap_newtype_ref {
    (
        $(#[$attr:meta])*
        $newtype:ident
    ) => {
        impl<'target> AsRef<Bitmap> for BitmapRef<'_, $newtype> {
            fn as_ref(&self) -> &Bitmap {
                let newtype: &$newtype = self.as_ref();
                newtype.as_ref()
            }
        }

        // NOTE: This seemingly useless impl is needed in order to have impls of
        //       IntoIterator<Item=BitmapIndex> for &BitmapRef<$newtype>
        #[doc(hidden)]
        impl Borrow<Bitmap> for &$newtype {
            fn borrow(&self) -> &Bitmap {
                self.as_ref()
            }
        }

        impl<'target> Borrow<Bitmap> for BitmapRef<'_, $newtype> {
            fn borrow(&self) -> &Bitmap {
                self.as_ref()
            }
        }

        impl<'target> From<BitmapRef<'target, Bitmap>> for BitmapRef<'target, $newtype> {
            fn from(input: BitmapRef<'target, Bitmap>) -> Self {
                input.cast()
            }
        }

        impl<'target> From<BitmapRef<'target, $newtype>> for BitmapRef<'target, Bitmap> {
            fn from(input: BitmapRef<'target, $newtype>) -> Self {
                input.cast()
            }
        }
    };
}

/// `BitmapRef` related tests for a specialized bitmap type
#[macro_export]
#[doc(hidden)]
macro_rules! impl_bitmap_newtype_ref_tests {
    (
        $(#[$attr:meta])*
        $newtype:ident
    ) => {
        // Check that newtypes keep implementing all expected traits,
        // in the interest of detecting future semver-breaking changes
        assert_impl_all!($newtype:
            BitAnd<BitmapRef<'static, $newtype>>,
            BitAnd<&'static BitmapRef<'static, $newtype>>,
            BitAndAssign<BitmapRef<'static, $newtype>>,
            BitAndAssign<&'static BitmapRef<'static, $newtype>>,
            BitOr<BitmapRef<'static, $newtype>>,
            BitOr<&'static BitmapRef<'static, $newtype>>,
            BitOrAssign<BitmapRef<'static, $newtype>>,
            BitOrAssign<&'static BitmapRef<'static, $newtype>>,
            BitXor<BitmapRef<'static, $newtype>>,
            BitXor<&'static BitmapRef<'static, $newtype>>,
            BitXorAssign<BitmapRef<'static, $newtype>>,
            BitXorAssign<&'static BitmapRef<'static, $newtype>>,
            PartialEq<BitmapRef<'static, $newtype>>,
            PartialEq<&'static BitmapRef<'static, $newtype>>,
            PartialOrd<BitmapRef<'static, $newtype>>,
            PartialOrd<&'static BitmapRef<'static, $newtype>>,
            Sub<BitmapRef<'static, $newtype>>,
            Sub<&'static BitmapRef<'static, $newtype>>,
            SubAssign<BitmapRef<'static, $newtype>>,
            SubAssign<&'static BitmapRef<'static, $newtype>>,
        );
        assert_impl_all!(&$newtype:
            BitAnd<BitmapRef<'static, $newtype>>,
            BitAnd<&'static BitmapRef<'static, $newtype>>,
            BitOr<BitmapRef<'static, $newtype>>,
            BitOr<&'static BitmapRef<'static, $newtype>>,
            BitXor<BitmapRef<'static, $newtype>>,
            BitXor<&'static BitmapRef<'static, $newtype>>,
            Sub<BitmapRef<'static, $newtype>>,
            Sub<&'static BitmapRef<'static, $newtype>>,
        );
        assert_impl_all!(BitmapRef<'static, $newtype>:
            AsRef<$newtype>, AsRef<Bitmap>,
            BitAnd<$newtype>, BitAnd<&'static $newtype>,
            BitAnd<BitmapRef<'static, $newtype>>,
            BitAnd<&'static BitmapRef<'static, $newtype>>,
            BitOr<$newtype>, BitOr<&'static $newtype>,
            BitOr<BitmapRef<'static, $newtype>>,
            BitOr<&'static BitmapRef<'static, $newtype>>,
            BitXor<$newtype>, BitXor<&'static $newtype>,
            BitXor<BitmapRef<'static, $newtype>>,
            BitXor<&'static BitmapRef<'static, $newtype>>,
            Borrow<$newtype>, Borrow<Bitmap>, Copy, Debug,
            Deref<Target=$newtype>, Display, Eq, From<&'static $newtype>,
            Hash, Into<BitmapRef<'static, Bitmap>>,
            IntoIterator<Item=BitmapIndex>, Not<Output=$newtype>, Ord,
            PartialEq<&'static $newtype>,
            PartialEq<BitmapRef<'static, $newtype>>,
            PartialEq<&'static BitmapRef<'static, $newtype>>,
            PartialOrd<&'static $newtype>,
            PartialOrd<BitmapRef<'static, $newtype>>,
            PartialOrd<&'static BitmapRef<'static, $newtype>>,
            Pointer, Sized, SpecializedBitmapRef<Owned=$newtype>,
            Sub<$newtype>, Sub<&'static $newtype>,
            Sub<BitmapRef<'static, $newtype>>,
            Sub<&'static BitmapRef<'static, $newtype>>,
            Sync, Unpin, UnwindSafe
        );
        assert_not_impl_any!(BitmapRef<'_, $newtype>:
            Binary, Default, Drop, Error, LowerExp, LowerHex, Octal, Read,
            UpperExp, UpperHex, fmt::Write, io::Write
        );
        assert_impl_all!(&BitmapRef<'static, $newtype>:
            BitAnd<$newtype>, BitAnd<&'static $newtype>,
            BitAnd<BitmapRef<'static, $newtype>>,
            BitAnd<&'static BitmapRef<'static, $newtype>>,
            BitOr<$newtype>, BitOr<&'static $newtype>,
            BitOr<BitmapRef<'static, $newtype>>,
            BitOr<&'static BitmapRef<'static, $newtype>>,
            BitXor<$newtype>, BitXor<&'static $newtype>,
            BitXor<BitmapRef<'static, $newtype>>,
            BitXor<&'static BitmapRef<'static, $newtype>>,
            IntoIterator<Item=BitmapIndex>,
            Not<Output=$newtype>,
            Sub<$newtype>, Sub<&'static $newtype>,
            Sub<BitmapRef<'static, $newtype>>,
            Sub<&'static BitmapRef<'static, $newtype>>,
        );

        #[test]
        fn static_checks_newtype() {
            assert_eq!(
                <$newtype as SpecializedBitmap>::BITMAP_KIND,
                BitmapKind::$newtype
            );
        }

        /// Test properties of [`BitmapRef`]s of bitmap newtypes
        fn test_newtype_ref_unary(
            new: &$newtype,
            new_ref: BitmapRef<'_, $newtype>
        ) -> Result<(), TestCaseError> {
            let clone: $newtype = new_ref.clone_target();
            prop_assert_eq!(clone, new);

            prop_assert_eq!(
                <BitmapRef<'_, _> as AsRef<$newtype>>::as_ref(&new_ref).as_ptr(),
                new.as_ptr()
            );
            prop_assert_eq!(
                <BitmapRef<'_, _> as AsRef<Bitmap>>::as_ref(&new_ref).as_ptr(),
                new.as_ptr()
            );
            prop_assert_eq!(
                <BitmapRef<'_, _> as Borrow<$newtype>>::borrow(&new_ref).as_ptr(),
                new.as_ptr()
            );
            prop_assert_eq!(
                <BitmapRef<'_, _> as Borrow<Bitmap>>::borrow(&new_ref).as_ptr(),
                new.as_ptr()
            );
            prop_assert_eq!(
                <&BitmapRef<'_, _> as Borrow<$newtype>>::borrow(&&new_ref).as_ptr(),
                new.as_ptr()
            );
            prop_assert_eq!(
                <BitmapRef<'_, _> as Deref>::deref(&new_ref).as_ptr(),
                new.as_ptr()
            );
            let bitmap_ref = BitmapRef::<Bitmap>::from(new_ref);
            prop_assert_eq!(bitmap_ref.as_ptr(), new_ref.as_ptr());
            let new_ref2 = BitmapRef::<$newtype>::from(bitmap_ref);
            prop_assert_eq!(new_ref2.as_ptr(), new_ref.as_ptr());

            prop_assert_eq!(format!("{new:?}"), format!("{new_ref:?}"));
            prop_assert_eq!(new.to_string(), new_ref.to_string());
            let state = RandomState::new();
            prop_assert_eq!(state.hash_one(new), state.hash_one(new_ref));
            prop_assert_eq!(format!("{:p}", new.as_ptr()), format!("{new_ref:p}"));

            prop_assert!(new
                .iter_set()
                .take(INFINITE_EXPLORE_ITERS)
                .eq(new_ref.into_iter().take(INFINITE_EXPLORE_ITERS)));
            prop_assert!(new
                .iter_set()
                .take(INFINITE_EXPLORE_ITERS)
                .eq((&new_ref).into_iter().take(INFINITE_EXPLORE_ITERS)));

            prop_assert_eq!(!new, !new_ref);
            prop_assert_eq!(!new, !&new_ref);
            Ok(())
        }

        /// Test binary operations involving [`BitmapRef`]
        fn test_newtype_ref_binops([new1, new2]: [$newtype; 2]) -> Result<(), TestCaseError> {
            let new1_ref = BitmapRef::from(&new1);
            prop_assert_eq!(new1_ref & &new2, &new1 & &new2);
            prop_assert_eq!(&new1_ref & &new2, &new1 & &new2);
            prop_assert_eq!(new1_ref | &new2, &new1 | &new2);
            prop_assert_eq!(&new1_ref | &new2, &new1 | &new2);
            prop_assert_eq!(new1_ref ^ &new2, &new1 ^ &new2);
            prop_assert_eq!(&new1_ref ^ &new2, &new1 ^ &new2);
            prop_assert_eq!(new1_ref - &new2, &new1 - &new2);
            prop_assert_eq!(&new1_ref - &new2, &new1 - &new2);
            prop_assert_eq!(new1_ref == &new2, &new1 == &new2);
            prop_assert_eq!(new1_ref.partial_cmp(&new2), new1.partial_cmp(&new2));
            prop_assert_eq!(new1_ref.cmp(&BitmapRef::from(&new2)), new1.cmp(&new2));
            Ok(())
        }
    };
}

#[allow(clippy::cognitive_complexity, clippy::op_ref, clippy::too_many_lines)]
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::bitmap::tests::INFINITE_EXPLORE_ITERS;
    use proptest::prelude::*;
    #[allow(unused)]
    use similar_asserts::assert_eq;
    use static_assertions::{
        assert_eq_align, assert_eq_size, assert_impl_all, assert_not_impl_any,
    };
    use std::{
        collections::hash_map::RandomState,
        error::Error,
        fmt::{Binary, LowerExp, LowerHex, Octal, UpperExp, UpperHex},
        hash::BuildHasher,
        io::{self, Read},
        ops::{BitAndAssign, BitOrAssign, BitXorAssign, SubAssign},
        panic::UnwindSafe,
    };

    // Check that public types in this module keep implementing all expected
    // traits, in the interest of detecting future semver-breaking changes
    assert_impl_all!(Bitmap:
        BitAnd<BitmapRef<'static, Bitmap>>,
        BitAnd<&'static BitmapRef<'static, Bitmap>>,
        BitAndAssign<BitmapRef<'static, Bitmap>>,
        BitAndAssign<&'static BitmapRef<'static, Bitmap>>,
        BitOr<BitmapRef<'static, Bitmap>>,
        BitOr<&'static BitmapRef<'static, Bitmap>>,
        BitOrAssign<BitmapRef<'static, Bitmap>>,
        BitOrAssign<&'static BitmapRef<'static, Bitmap>>,
        BitXor<BitmapRef<'static, Bitmap>>,
        BitXor<&'static BitmapRef<'static, Bitmap>>,
        BitXorAssign<BitmapRef<'static, Bitmap>>,
        BitXorAssign<&'static BitmapRef<'static, Bitmap>>,
        PartialEq<BitmapRef<'static, Bitmap>>,
        PartialEq<&'static BitmapRef<'static, Bitmap>>,
        PartialOrd<BitmapRef<'static, Bitmap>>,
        PartialOrd<&'static BitmapRef<'static, Bitmap>>,
        Sub<BitmapRef<'static, Bitmap>>,
        Sub<&'static BitmapRef<'static, Bitmap>>,
        SubAssign<BitmapRef<'static, Bitmap>>,
        SubAssign<&'static BitmapRef<'static, Bitmap>>,
    );
    assert_impl_all!(&Bitmap:
        BitAnd<BitmapRef<'static, Bitmap>>,
        BitAnd<&'static BitmapRef<'static, Bitmap>>,
        BitOr<BitmapRef<'static, Bitmap>>,
        BitOr<&'static BitmapRef<'static, Bitmap>>,
        BitXor<BitmapRef<'static, Bitmap>>,
        BitXor<&'static BitmapRef<'static, Bitmap>>,
        Sub<BitmapRef<'static, Bitmap>>,
        Sub<&'static BitmapRef<'static, Bitmap>>,
    );
    assert_eq_align!(BitmapRef<'static, Bitmap>, NonNull<hwloc_bitmap_s>);
    assert_eq_size!(BitmapRef<'static, Bitmap>, NonNull<hwloc_bitmap_s>);
    assert_impl_all!(BitmapRef<'static, Bitmap>:
        AsRef<Bitmap>,
        BitAnd<Bitmap>, BitAnd<&'static Bitmap>,
        BitAnd<BitmapRef<'static, Bitmap>>,
        BitAnd<&'static BitmapRef<'static, Bitmap>>,
        BitOr<Bitmap>, BitOr<&'static Bitmap>,
        BitOr<BitmapRef<'static, Bitmap>>,
        BitOr<&'static BitmapRef<'static, Bitmap>>,
        BitXor<Bitmap>, BitXor<&'static Bitmap>,
        BitXor<BitmapRef<'static, Bitmap>>,
        BitXor<&'static BitmapRef<'static, Bitmap>>,
        Borrow<Bitmap>, Copy, Debug, Deref<Target=Bitmap>, Display,
        From<&'static Bitmap>, Hash, IntoIterator<Item=BitmapIndex>,
        Not<Output=Bitmap>, Ord,
        PartialEq<&'static Bitmap>,
        PartialEq<BitmapRef<'static, Bitmap>>,
        PartialEq<&'static BitmapRef<'static, Bitmap>>,
        PartialOrd<&'static Bitmap>,
        PartialOrd<BitmapRef<'static, Bitmap>>,
        PartialOrd<&'static BitmapRef<'static, Bitmap>>,
        Pointer, Sized,
        Sub<Bitmap>, Sub<&'static Bitmap>,
        Sub<BitmapRef<'static, Bitmap>>,
        Sub<&'static BitmapRef<'static, Bitmap>>,
        Sync, Unpin, UnwindSafe
    );
    assert_not_impl_any!(BitmapRef<'_, Bitmap>:
        Binary, Default, Drop, Error, LowerExp, LowerHex, Octal, Read, UpperExp,
        UpperHex, fmt::Write, io::Write
    );
    assert_impl_all!(&BitmapRef<'static, Bitmap>:
        BitAnd<Bitmap>, BitAnd<&'static Bitmap>,
        BitAnd<BitmapRef<'static, Bitmap>>,
        BitAnd<&'static BitmapRef<'static, Bitmap>>,
        BitOr<Bitmap>, BitOr<&'static Bitmap>,
        BitOr<BitmapRef<'static, Bitmap>>,
        BitOr<&'static BitmapRef<'static, Bitmap>>,
        BitXor<Bitmap>, BitXor<&'static Bitmap>,
        BitXor<BitmapRef<'static, Bitmap>>,
        BitXor<&'static BitmapRef<'static, Bitmap>>,
        IntoIterator<Item=BitmapIndex>,
        Not<Output=Bitmap>,
        Sub<Bitmap>, Sub<&'static Bitmap>,
        Sub<BitmapRef<'static, Bitmap>>,
        Sub<&'static BitmapRef<'static, Bitmap>>,
    );

    /// Check a [`BitmapRef`] against the [`Bitmap`] it's supposed to refer to
    pub(crate) fn test_bitmap_ref_unary(
        bitmap: &Bitmap,
        bitmap_ref: BitmapRef<'_, Bitmap>,
    ) -> Result<(), TestCaseError> {
        let clone: Bitmap = bitmap_ref.clone_target();
        prop_assert_eq!(clone, bitmap);

        prop_assert_eq!(
            <BitmapRef<'_, _> as AsRef<Bitmap>>::as_ref(&bitmap_ref).as_ptr(),
            bitmap.as_ptr()
        );
        prop_assert_eq!(
            <BitmapRef<'_, _> as Borrow<Bitmap>>::borrow(&bitmap_ref).as_ptr(),
            bitmap.as_ptr()
        );
        prop_assert_eq!(
            <&BitmapRef<'_, _> as Borrow<Bitmap>>::borrow(&&bitmap_ref).as_ptr(),
            bitmap.as_ptr()
        );
        prop_assert_eq!(
            <BitmapRef<'_, _> as Deref>::deref(&bitmap_ref).as_ptr(),
            bitmap.as_ptr()
        );

        prop_assert_eq!(format!("{bitmap:?}"), format!("{bitmap_ref:?}"));
        prop_assert_eq!(bitmap.to_string(), bitmap_ref.to_string());
        let state = RandomState::new();
        prop_assert_eq!(state.hash_one(bitmap), state.hash_one(bitmap_ref));
        prop_assert_eq!(format!("{:p}", bitmap.0), format!("{bitmap_ref:p}"));

        prop_assert!(bitmap
            .iter_set()
            .take(INFINITE_EXPLORE_ITERS)
            .eq(bitmap_ref.into_iter().take(INFINITE_EXPLORE_ITERS)));
        prop_assert!(bitmap
            .iter_set()
            .take(INFINITE_EXPLORE_ITERS)
            .eq((&bitmap_ref).into_iter().take(INFINITE_EXPLORE_ITERS)));

        prop_assert_eq!(!bitmap, !bitmap_ref);
        prop_assert_eq!(!bitmap, !&bitmap_ref);
        Ok(())
    }

    /// Check binary operations of [`BitmapRef`]
    pub(crate) fn test_bitmap_ref_binops(
        bitmap: &Bitmap,
        other: &Bitmap,
    ) -> Result<(), TestCaseError> {
        let bitmap_ref = BitmapRef::from(bitmap);

        prop_assert_eq!(bitmap_ref & other, bitmap & other);
        prop_assert_eq!(&bitmap_ref & other, bitmap & other);
        prop_assert_eq!(bitmap_ref | other, bitmap | other);
        prop_assert_eq!(&bitmap_ref | other, bitmap | other);
        prop_assert_eq!(bitmap_ref ^ other, bitmap ^ other);
        prop_assert_eq!(&bitmap_ref ^ other, bitmap ^ other);
        prop_assert_eq!(bitmap_ref - other, bitmap - other);
        prop_assert_eq!(&bitmap_ref - other, bitmap - other);

        prop_assert_eq!(bitmap_ref == other, bitmap == other);
        prop_assert_eq!(bitmap_ref.partial_cmp(other), bitmap.partial_cmp(other));
        prop_assert_eq!(bitmap_ref.cmp(&BitmapRef::from(other)), bitmap.cmp(other));
        Ok(())
    }
}
